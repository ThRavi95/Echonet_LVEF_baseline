"""
Spatiotemporal Explainability Engine (Grad-CAM 3D)
===================================================
Implementation of 3D Gradient-weighted Class Activation Mapping (Grad-CAM)
tailored for continuous LVEF regression using spatiotemporal 3D CNNs
such as R(2+1)D-18 (Ouyang et al., Nature 2020).

Key Capabilities:
  • Hooks into intermediate 3D convolutional blocks (e.g. layer4 / last conv block)
  • Global spatiotemporal average pooling of 3D gradients: alpha_k = (1 / (T*H*W)) * sum(dY/dA_k)
  • Rectified linear activation combination: L_GradCAM = ReLU(sum(alpha_k * A_k))
  • Trilinear spatiotemporal interpolation back to the original video resolution (32x112x112)
  • Dynamic min-max normalization to [0.0, 1.0]
  • Resource-safe lifecycle with explicit hook removal and context manager support
"""
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM3D:
    """
    3D Grad-CAM for spatiotemporal continuous regression models.

    Computes activation heatmaps by projecting gradients from a scalar
    continuous prediction (e.g. LVEF %) back to the final 3D convolutional layer.

    Args:
        model: Trained PyTorch 3D CNN / spatiotemporal model (e.g. EchoNetR2Plus1D).
        target_layer: The target convolutional layer/module to hook into. If None,
                      automatically resolves `model.layer4`, `model.backbone.layer4`,
                      or calls `model.get_last_conv_layer()`.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[Union[nn.Module, str]] = None,
    ) -> None:
        self.model = model
        self.target_layer = self._resolve_target_layer(target_layer)

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._forward_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._backward_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        self._register_hooks()

    def _resolve_target_layer(
        self, target_layer: Optional[Union[nn.Module, str]]
    ) -> nn.Module:
        """Resolve the target module from explicit reference, string path, or model defaults."""
        if isinstance(target_layer, nn.Module):
            return target_layer

        if isinstance(target_layer, str):
            curr: Any = self.model
            for part in target_layer.split("."):
                if not hasattr(curr, part):
                    raise AttributeError(
                        f"Model does not contain submodule path '{target_layer}' (failed at '{part}')."
                    )
                curr = getattr(curr, part)
            if not isinstance(curr, nn.Module):
                raise TypeError(f"Resolved target '{target_layer}' is not an nn.Module.")
            return curr

        # Automatic resolution for EchoNet architectures
        if hasattr(self.model, "get_last_conv_layer") and callable(self.model.get_last_conv_layer):
            last_conv = self.model.get_last_conv_layer()
            if isinstance(last_conv, nn.Module):
                return last_conv

        if hasattr(self.model, "layer4") and isinstance(self.model.layer4, nn.Module):
            return self.model.layer4

        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "layer4"):
            layer4 = getattr(self.model.backbone, "layer4")
            if isinstance(layer4, nn.Module):
                return layer4

        raise ValueError(
            "Could not automatically resolve target_layer. Please pass an explicit "
            "nn.Module (e.g. model.backbone.layer4[-1] or model.layer4)."
        )

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        def _forward_hook(module: nn.Module, input_tensor: Any, output_tensor: torch.Tensor) -> None:
            # Shape: (B, C, T', H', W')
            self.activations = output_tensor.detach()

        def _backward_hook(
            module: nn.Module, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]
        ) -> None:
            # Shape: (B, C, T', H', W')
            self.gradients = grad_output[0].detach()

        self._forward_handle = self.target_layer.register_forward_hook(_forward_hook)

        if hasattr(self.target_layer, "register_full_backward_hook"):
            self._backward_handle = self.target_layer.register_full_backward_hook(_backward_hook)
        else:
            self._backward_handle = self.target_layer.register_backward_hook(_backward_hook)

    def remove_hooks(self) -> None:
        """Remove active forward and backward hooks to prevent memory leaks."""
        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None
        if self._backward_handle is not None:
            self._backward_handle.remove()
            self._backward_handle = None

    def __enter__(self) -> "GradCAM3D":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.remove_hooks()

    def __del__(self) -> None:
        self.remove_hooks()

    def generate_heatmap(
        self,
        video: torch.Tensor,
        target_value: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute the 3D Grad-CAM heatmap for a single video clip.

        Args:
            video: Input video tensor of shape (1, 3, T, H, W) or (3, T, H, W).
            target_value: Optional scalar override. If None, the model's forward
                          prediction (scalar LVEF) is used as the backward target.

        Returns:
            Normalized 3D heatmap tensor of shape (T, H, W) with values in [0.0, 1.0].
        """
        heatmap, _ = self.generate_heatmap_with_prediction(video, target_value=target_value)
        return heatmap

    def generate_heatmap_with_prediction(
        self,
        video: torch.Tensor,
        target_value: Optional[float] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute the 3D Grad-CAM heatmap along with the predicted scalar LVEF.

        Args:
            video: Input video tensor of shape (1, 3, T, H, W) or (3, T, H, W).
            target_value: Optional target scalar for backward differentiation.

        Returns:
            Tuple of:
              - heatmap: torch.Tensor of shape (T, H, W) normalized to [0.0, 1.0]
              - prediction: float scalar representing the predicted LVEF %
        """
        if video.dim() == 4:
            # (3, T, H, W) -> (1, 3, T, H, W)
            video = video.unsqueeze(0)

        if video.dim() != 5 or video.shape[1] != 3:
            raise ValueError(
                f"Expected video tensor of shape (1, 3, T, H, W) or (3, T, H, W), "
                f"got shape {tuple(video.shape)}."
            )

        batch_size, _, orig_t, orig_h, orig_w = video.shape
        if batch_size != 1:
            raise ValueError(
                f"GradCAM3D currently supports single-sample generation (batch_size=1), "
                f"got batch_size={batch_size}."
            )

        was_training = self.model.training
        self.model.eval()

        self.activations = None
        self.gradients = None

        with torch.enable_grad():
            self.model.zero_grad()

            device = next(self.model.parameters()).device
            video_input = video.to(device)

            pred = self.model(video_input)
            pred_scalar = pred.view(-1)[0]
            predicted_ef_value = float(pred_scalar.item())

            # Backward pass on the scalar regression output
            if target_value is not None:
                backward_target = torch.tensor(target_value, device=device, dtype=torch.float32)
                loss = F.mse_loss(pred_scalar, backward_target)
                loss.backward(retain_graph=False)
            else:
                pred_scalar.backward(retain_graph=False)

        if was_training:
            self.model.train()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "GradCAM3D hooks failed to capture activations or gradients. "
                "Ensure target_layer participates in the forward and backward computation graph."
            )

        # Activations & Gradients: shape (1, C, T', H', W')
        activations = self.activations
        gradients = self.gradients

        # 1. Global average pooling of gradients across spatiotemporal dimensions (T', H', W')
        # weights shape: (1, C, 1, 1, 1)
        weights = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)

        # 2. Weighted combination of forward activation maps across channel dimension C
        # cam shape: (1, 1, T', H', W')
        cam = torch.sum(weights * activations, dim=1, keepdim=True)

        # 3. Apply ReLU to retain features that positively contribute to prediction
        cam = F.relu(cam)

        # 4. Spatiotemporal trilinear upsampling back to original input resolution (T, H, W)
        cam = F.interpolate(
            cam,
            size=(orig_t, orig_h, orig_w),
            mode="trilinear",
            align_corners=False,
        )

        # 5. Normalization to [0.0, 1.0]
        cam = cam.squeeze(0).squeeze(0)  # (T, H, W)
        cam_min = cam.min()
        cam_max = cam.max()

        if (cam_max - cam_min) > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.detach(), predicted_ef_value

    @staticmethod
    def overlay_heatmap_on_video(
        video_frames: np.ndarray,
        heatmap: Union[torch.Tensor, np.ndarray],
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> np.ndarray:
        """
        Overlay a 3D Grad-CAM heatmap onto RGB video frames for visualization.

        Args:
            video_frames: (T, H, W, 3) uint8 or float32 [0, 1] array of echocardiography frames.
            heatmap: (T, H, W) float array or tensor in range [0.0, 1.0].
            alpha: Blending weight for heatmap (0.0 = only video, 1.0 = only heatmap).
            colormap: Matplotlib colormap name ('jet', 'viridis', 'hot', etc.).

        Returns:
            (T, H, W, 3) uint8 array of blended RGB frames ready for video export.
        """
        import matplotlib.cm as cm

        if isinstance(heatmap, torch.Tensor):
            heatmap_np = heatmap.detach().cpu().numpy()
        else:
            heatmap_np = np.asarray(heatmap)

        video_np = np.asarray(video_frames)
        if video_np.dtype != np.uint8:
            if video_np.max() <= 1.0:
                video_np = (video_np * 255.0).clip(0, 255).astype(np.uint8)
            else:
                video_np = video_np.clip(0, 255).astype(np.uint8)

        t_dim, h_dim, w_dim = heatmap_np.shape
        cmap = cm.get_cmap(colormap)

        blended_frames = []
        for t in range(t_dim):
            frame_base = video_np[t].astype(np.float32)
            cam_frame = heatmap_np[t]

            # Apply colormap -> RGBA [0.0, 1.0] -> RGB [0, 255]
            cam_colored = (cmap(cam_frame)[:, :, :3] * 255.0).astype(np.float32)

            blended = (1.0 - alpha) * frame_base + alpha * cam_colored
            blended_frames.append(np.clip(blended, 0, 255).astype(np.uint8))

        return np.stack(blended_frames, axis=0)

    @staticmethod
    def compute_chamber_attention_fraction(
        heatmap_frame: Union[torch.Tensor, np.ndarray],
        chamber_mask: Union[torch.Tensor, np.ndarray],
        attention_threshold: float = 0.5,
    ) -> float:
        """
        Quantify the fraction of salient Grad-CAM attention located inside
        the ground-truth Left Ventricular (LV) chamber mask (XAI pointing game / fidelity).

        Args:
            heatmap_frame: (H, W) Grad-CAM heatmap in [0.0, 1.0].
            chamber_mask: (H, W) Binary ground-truth LV segmentation mask (1 = inside LV, 0 = outside).
            attention_threshold: Minimum heatmap value to qualify as salient attention.

        Returns:
            Fraction in [0.0, 1.0] representing (attention_inside_LV / total_salient_attention).
        """
        if isinstance(heatmap_frame, torch.Tensor):
            heatmap_frame = heatmap_frame.detach().cpu().numpy()
        if isinstance(chamber_mask, torch.Tensor):
            chamber_mask = chamber_mask.detach().cpu().numpy()

        salient = (heatmap_frame >= attention_threshold).astype(np.float32)
        total_salient = float(salient.sum())

        if total_salient < 1e-6:
            return 0.0

        mask_binary = (chamber_mask > 0).astype(np.float32)
        salient_inside_chamber = float((salient * mask_binary).sum())

        return salient_inside_chamber / total_salient

