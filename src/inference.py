"""
EchoNet-Dynamic LVEF Inference & Explainability Pipeline
========================================================
Provides deterministic multi-clip inference, 3D Grad-CAM spatiotemporal
explainability, and automated clinical diagnostic report generation
from a trained R(2+1)D-18 checkpoint.

No synthetic fallbacks. A FileNotFoundError is raised if the checkpoint is absent.
"""
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.clinical.report import ClinicalReportGenerator
from src.config import VideoConfig
from src.data.io import load_multiclip_video, load_video_clip
from src.metrics.explainability import GradCAM3D
from src.models.r2plus1d import EchoNetR2Plus1D


class LVEFEstimator:
    """
    Inference wrapper for the trained R(2+1)D-18 LVEF regression model.

    Uses multi-clip averaging (4 evenly-spaced clips) for robust prediction,
    and 3D Grad-CAM for spatiotemporal attention attribution.

    Args:
        checkpoint_path: Path to the trained .pt checkpoint file.
                         Raises FileNotFoundError if the file does not exist.
        device:          Force a specific device ('cpu', 'cuda', 'mps').
                         Defaults to CUDA if available, else CPU.
        num_clips:       Number of evenly-spaced clips to average (default 4).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        num_clips: int = 4,
    ):
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Trained model checkpoint not found: '{checkpoint_path}'\n\n"
                "To obtain a checkpoint:\n"
                "  1. Run training on Kaggle:\n"
                "       python kaggle/train_kaggle.py \\\n"
                "           --data_dir /kaggle/input/echonet-dynamic/EchoNet-Dynamic \\\n"
                "           --output_dir /kaggle/working/outputs\n"
                "  2. Download the saved best_r2plus1d_lvef.pt from Kaggle output.\n"
                "  3. Place it at: outputs/best_r2plus1d_lvef.pt"
            )

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.cfg = VideoConfig()
        self.num_clips = num_clips
        self.report_generator = ClinicalReportGenerator()

        self.model = EchoNetR2Plus1D(pretrained=False, dropout_rate=0.0)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        # Strip DataParallel module prefixes if present
        clean_state = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state.items()
        }

        self.model.load_state_dict(clean_state)
        self.model.to(self.device)
        self.model.eval()
        print(f"[LVEFEstimator] Checkpoint loaded successfully: {checkpoint_path}")

    @torch.no_grad()
    def predict(self, video_path: str) -> Dict[str, Any]:
        """
        Predict LVEF from a single video file using multi-clip averaging.

        Args:
            video_path: Path to an .avi echocardiogram (or .npy companion).

        Returns:
            Dict with keys: predicted_lvef, raw_output, clinical_category,
            badge_color, ase_classification, summary, report.
        """
        if not os.path.exists(video_path) and not os.path.exists(
            video_path.replace(".avi", ".npy")
        ):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # (num_clips, T, H, W, 3) float32 normalised
        multi_clip = load_multiclip_video(
            video_path,
            num_clips=self.num_clips,
            clip_len=self.cfg.num_frames,
            stride=self.cfg.frame_stride,
            target_size=self.cfg.image_size,
            mean=self.cfg.mean,
            std=self.cfg.std,
        )

        # -> (num_clips, 3, T, H, W)
        clips_tensor = torch.from_numpy(
            np.ascontiguousarray(multi_clip.transpose(0, 4, 1, 2, 3))
        ).float().to(self.device)

        preds = self.model(clips_tensor)        # (num_clips,)
        raw_pred = float(preds.mean().item())
        clamped_ef = float(np.clip(raw_pred, 10.0, 85.0))

        report_dict = self.report_generator.generate_report(
            patient_id=os.path.basename(video_path),
            predicted_ef=clamped_ef,
        )

        classification = report_dict["classification"]
        return {
            "predicted_lvef":     round(clamped_ef, 1),
            "raw_output":         round(raw_pred, 2),
            "heart_risk_level":   classification.get("heart_risk_level", "Undetermined"),
            "clinical_category":  classification["phenotype"],
            "badge_color":        classification["badge_color"],
            "ase_classification": classification["category"],
            "summary":            report_dict["narrative"],
            "report":             report_dict,
        }

    def explain(
        self,
        video_path: str,
        start_frame: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate 3D Grad-CAM spatiotemporal heatmap for a 32-frame clip of the video.

        Args:
            video_path: Path to video file.
            start_frame: Starting frame index for the 32-frame clip.

        Returns:
            Tuple of:
              - raw_frames: (32, 112, 112, 3) float32 in range [0, 1]
              - heatmap_3d: (32, 112, 112) float32 normalized in range [0, 1]
              - predicted_ef: scalar predicted LVEF % for this clip
        """
        clip_np = load_video_clip(
            video_path=video_path,
            start_frame=start_frame,
            num_frames=self.cfg.num_frames,
            stride=self.cfg.frame_stride,
            target_size=self.cfg.image_size,
            mean=self.cfg.mean,
            std=self.cfg.std,
        )  # (32, 112, 112, 3)

        # Un-normalized display frames in [0, 1]
        raw_frames = np.clip(clip_np * self.cfg.std + self.cfg.mean, 0.0, 1.0)

        # Tensor for model: (1, 3, 32, 112, 112)
        video_tensor = torch.from_numpy(clip_np).permute(3, 0, 1, 2).unsqueeze(0).float().to(self.device)

        grad_cam = GradCAM3D(model=self.model, target_layer=self.model.layer4)
        try:
            heatmap_3d, pred_ef = grad_cam.generate_heatmap_with_prediction(video_tensor)
        finally:
            grad_cam.remove_hooks()

        return raw_frames, heatmap_3d.cpu().numpy(), float(pred_ef)
