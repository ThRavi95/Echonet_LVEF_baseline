"""
Ultrasound Video Degradation & Stress Simulation Engine
=======================================================
Simulates physical and electronic ultrasound imaging artifacts common in
Point-of-Care Ultrasound (POCUS) and clinical echocardiography:

1. Speckle Noise (multiplicative Rayleigh / Gaussian interference scatter)
2. Acoustic Shadowing (sector/wedge signal attenuation from ribs, dense calcifications, or prostheses)
3. Frame Dropout (temporal frame freezes / blackouts from wireless transducer packet loss)
4. Compound Clinical POCUS Degradation (combined stress testing)

Used for Objective 1 and robustness benchmarking of LVEF regression and
XAI explanation fidelity under adverse acoustic conditions.
"""
from typing import Optional, Tuple, Union

import numpy as np
import torch


class UltrasoundDegradation:
    """
    Applies controlled echocardiographic degradations to video clips.

    Supports both NumPy arrays:
      - (T, H, W, C) or (T, H, W) in range [0, 1] or [0, 255]
    and PyTorch tensors:
      - (B, C, T, H, W) or (C, T, H, W) in range [0, 1]
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    # -----------------------------------------------------------------------
    # 1. Multiplicative Speckle Noise
    # -----------------------------------------------------------------------

    def add_speckle_noise(
        self,
        video: Union[np.ndarray, torch.Tensor],
        severity: float = 0.2,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply multiplicative speckle noise: I_noisy = I + I * eta,
        where eta ~ N(0, severity^2).

        Args:
            video: Input video array or tensor.
            severity: Noise standard deviation (0.0 = none, 0.2 = moderate, 0.4 = severe).

        Returns:
            Degraded video with identical shape and data type.
        """
        is_torch = isinstance(video, torch.Tensor)
        device = video.device if is_torch else None
        dtype = video.dtype if is_torch else None

        arr = video.detach().cpu().numpy() if is_torch else np.array(video, copy=True)
        is_uint8 = arr.dtype == np.uint8

        if is_uint8:
            arr_float = arr.astype(np.float32) / 255.0
        else:
            arr_float = arr.astype(np.float32)

        # Multiplicative Gaussian noise
        noise = self.rng.normal(loc=0.0, scale=severity, size=arr_float.shape).astype(np.float32)
        noisy = arr_float + arr_float * noise
        noisy = np.clip(noisy, 0.0, 1.0)

        if is_uint8:
            result = (noisy * 255.0).astype(np.uint8)
        else:
            result = noisy.astype(np.float32)

        if is_torch:
            return torch.from_numpy(result).to(device=device, dtype=dtype)
        return result

    # -----------------------------------------------------------------------
    # 2. Acoustic Shadowing Artifact
    # -----------------------------------------------------------------------

    def apply_acoustic_shadowing(
        self,
        video: Union[np.ndarray, torch.Tensor],
        attenuation_factor: float = 0.75,
        center_angle_deg: Optional[float] = None,
        angular_width_deg: float = 25.0,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Simulate an acoustic shadow cone originating from the transducer apex.
        Attenuates acoustic signal within a radial sector by attenuation_factor.

        Args:
            video: Input video array (T, H, W, C) or tensor (B, C, T, H, W).
            attenuation_factor: Signal reduction ratio in [0.0, 1.0] (0.75 = 75% attenuated).
            center_angle_deg: Center of shadow cone in degrees (-45° to +45° from apex vertical).
                              If None, randomly selected between -30° and +30°.
            angular_width_deg: Angular width of shadow cone in degrees.

        Returns:
            Degraded video with acoustic shadowing.
        """
        is_torch = isinstance(video, torch.Tensor)
        device = video.device if is_torch else None
        dtype = video.dtype if is_torch else None

        arr = video.detach().cpu().numpy() if is_torch else np.array(video, copy=True)
        is_uint8 = arr.dtype == np.uint8

        # Extract spatial dimensions H, W
        if is_torch and arr.ndim == 5:
            # (B, C, T, H, W)
            H, W = arr.shape[3], arr.shape[4]
        elif arr.ndim >= 3:
            # (T, H, W, C) or (T, H, W)
            H, W = arr.shape[1], arr.shape[2]
        else:
            raise ValueError(f"Unsupported array shape for acoustic shadowing: {arr.shape}")

        if center_angle_deg is None:
            center_angle_deg = float(self.rng.uniform(-25.0, 25.0))

        # Build 2D radial attenuation mask
        # Apex is at top center: (x_apex, y_apex) = (W / 2, 0)
        apex_x = (W - 1) / 2.0
        apex_y = 0.0

        y_coords, x_coords = np.mgrid[0:H, 0:W]
        dx = x_coords - apex_x
        dy = y_coords - apex_y

        # Angle from vertical (downward positive y) in degrees
        angles_deg = np.degrees(np.arctan2(dx, dy + 1e-5))

        # Sector mask with smooth boundary transition
        half_width = angular_width_deg / 2.0
        angle_diff = np.abs(angles_deg - center_angle_deg)

        # Smooth sigmoidal feathering at shadow edges (width 3 degrees)
        feather_deg = 3.0
        inside_weight = np.clip((half_width - angle_diff) / feather_deg + 0.5, 0.0, 1.0)

        # Shadow mask: 1.0 outside shadow, (1.0 - attenuation_factor) deep inside shadow
        shadow_mask_2d = 1.0 - (inside_weight * attenuation_factor)
        shadow_mask_2d = shadow_mask_2d.astype(np.float32)

        # Broadcast shadow mask across dimensions
        if is_torch and arr.ndim == 5:
            # (1, 1, 1, H, W)
            mask_expanded = shadow_mask_2d[np.newaxis, np.newaxis, np.newaxis, :, :]
        elif arr.ndim == 4:
            # (T, H, W, C) -> (1, H, W, 1)
            mask_expanded = shadow_mask_2d[np.newaxis, :, :, np.newaxis]
        else:
            # (T, H, W) -> (1, H, W)
            mask_expanded = shadow_mask_2d[np.newaxis, :, :]

        if is_uint8:
            arr_float = arr.astype(np.float32)
            degraded = np.clip(arr_float * mask_expanded, 0.0, 255.0).astype(np.uint8)
        else:
            degraded = np.clip(arr * mask_expanded, 0.0, 1.0).astype(np.float32)

        if is_torch:
            return torch.from_numpy(degraded).to(device=device, dtype=dtype)
        return degraded

    # -----------------------------------------------------------------------
    # 3. Frame Dropout / Wireless Temporal Packet Loss
    # -----------------------------------------------------------------------

    def apply_frame_dropout(
        self,
        video: Union[np.ndarray, torch.Tensor],
        drop_rate: float = 0.25,
        mode: str = "freeze",
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Simulate temporal packet loss or frame drops during transmission.

        Args:
            video: Input video array or tensor.
            drop_rate: Proportion of frames dropped (0.1 to 0.4).
            mode: 'freeze' (replaces dropped frames with last valid frame) or
                  'black' (replaces dropped frames with zero / black).

        Returns:
            Degraded video with temporal frame dropout.
        """
        is_torch = isinstance(video, torch.Tensor)
        device = video.device if is_torch else None
        dtype = video.dtype if is_torch else None

        arr = video.detach().cpu().numpy() if is_torch else np.array(video, copy=True)

        if is_torch and arr.ndim == 5:
            # (B, C, T, H, W) -> temporal axis is 2
            T = arr.shape[2]
            drop_mask = self.rng.random(T) < drop_rate
            drop_mask[0] = False  # Keep first frame intact as reference

            for t in range(1, T):
                if drop_mask[t]:
                    if mode == "freeze":
                        arr[:, :, t] = arr[:, :, t - 1]
                    else:
                        arr[:, :, t] = 0.0
        elif arr.ndim >= 3:
            # (T, H, W, C) or (T, H, W) -> temporal axis is 0
            T = arr.shape[0]
            drop_mask = self.rng.random(T) < drop_rate
            drop_mask[0] = False

            for t in range(1, T):
                if drop_mask[t]:
                    if mode == "freeze":
                        arr[t] = arr[t - 1]
                    else:
                        arr[t] = 0.0

        if is_torch:
            return torch.from_numpy(arr).to(device=device, dtype=dtype)
        return arr

    # -----------------------------------------------------------------------
    # 4. Compound Clinical POCUS Degradation
    # -----------------------------------------------------------------------

    def apply_compound_degradation(
        self,
        video: Union[np.ndarray, torch.Tensor],
        severity: str = "moderate",
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply a realistic combination of speckle noise, acoustic shadowing,
        and temporal frame dropout to simulate bedside clinical scanning.

        Args:
            video: Input video.
            severity: 'mild', 'moderate', or 'severe'.

        Returns:
            Degraded video under compound stress conditions.
        """
        params = {
            "mild": {"speckle": 0.15, "shadow": 0.50, "drop": 0.10},
            "moderate": {"speckle": 0.25, "shadow": 0.75, "drop": 0.20},
            "severe": {"speckle": 0.40, "shadow": 0.90, "drop": 0.35},
        }.get(severity, {"speckle": 0.25, "shadow": 0.75, "drop": 0.20})

        degraded = self.add_speckle_noise(video, severity=params["speckle"])
        degraded = self.apply_acoustic_shadowing(degraded, attenuation_factor=params["shadow"])
        degraded = self.apply_frame_dropout(degraded, drop_rate=params["drop"], mode="freeze")
        return degraded

