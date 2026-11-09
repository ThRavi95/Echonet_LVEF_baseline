"""
EchoNet-Dynamic PyTorch Dataset
================================
Implements strict split-aware sampling, spatial augmentation, normalisation,
and optional cardiac phase mask generation aligned with Ouyang et al. (Nature 2020).

Augmentation strategy:
  TRAIN  — random temporal start + zero-pad 112→136 → consistent random 112-crop
  VAL    — deterministic centred clip, no spatial padding/cropping
  TEST   — centred clip (or multi-clip via load_multiclip_video at inference time)
"""
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import VideoConfig
from src.data.io import load_video_clip, get_video_total_frames
from src.data.parser import EchoNetParser
from src.data.rasterizer import rasterize_lv_mask


class EchoNetVideoDataset(Dataset):
    """
    PyTorch Dataset for EchoNet-Dynamic video-based LVEF regression.

    Output tensor shape: (3, num_frames, image_size, image_size) — (C, T, H, W).
    All normalisation is performed inside load_video_clip; this class only
    handles temporal start selection, spatial augmentation, and mask assembly.

    Args:
        df:               Split-specific metadata DataFrame from EchoNetParser.
        videos_dir:       Absolute path to the Videos/ folder.
        num_frames:       Temporal depth (default 32).
        frame_stride:     Raw-frame stride (default 2).
        image_size:       Final spatial resolution after crop (default 112).
        padding:          Zero-pad size on each spatial side before random crop
                          (applied only during training, default 12).
        is_train:         If True, apply random start + random crop augmentation.
        mean:             Normalisation mean ([0,1] domain).
        std:              Normalisation std  ([0,1] domain).
        parser:           EchoNetParser instance for cardiac phase lookup.
        return_phase_mask: If True, include binary ED/ES masks and clip indices.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        videos_dir: str,
        num_frames: int = 32,
        frame_stride: int = 2,
        image_size: int = 112,
        padding: int = 12,
        is_train: bool = True,
        mean: float = 0.12,
        std: float = 0.18,
        parser: Optional[EchoNetParser] = None,
        return_phase_mask: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.videos_dir = videos_dir
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.image_size = image_size
        self.padding = padding
        self.is_train = is_train
        self.mean = mean
        self.std = std
        self.parser = parser
        self.return_phase_mask = return_phase_mask

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _choose_start_frame(self, video_path: str) -> int:
        """
        Select the temporal start frame for clip extraction.

        TRAIN: uniform-random over [0, max_start] where max_start = max(0, T - span).
        VAL/TEST: deterministic centre, floor((T - span) / 2).
        """
        total = get_video_total_frames(video_path)
        span = self.num_frames * self.frame_stride  # raw frames consumed

        if total <= span:
            return 0  # will be padded inside load_video_clip

        max_start = total - span
        if self.is_train:
            return int(np.random.randint(0, max_start + 1))
        else:
            return int(max_start // 2)

    def _apply_spatial_augmentation(self, clip: np.ndarray) -> np.ndarray:
        """
        Zero-pad to (image_size + 2*padding) × (image_size + 2*padding),
        then take a random 112×112 crop consistently across all T frames.

        Args:
            clip: float32 array (T, H, W, C).

        Returns:
            Cropped float32 array (T, image_size, image_size, C).
        """
        pad = self.padding
        padded_size = self.image_size + 2 * pad   # e.g. 112 + 24 = 136

        # np.pad pads spatial dims: (before, after) for axes (H, W)
        # clip axes: (T, H, W, C) → pad axes 1 and 2
        clip_padded = np.pad(
            clip,
            pad_width=[(0, 0), (pad, pad), (pad, pad), (0, 0)],
            mode="constant",
            constant_values=0.0,
        )  # (T, 136, 136, C)

        # Sample one random crop origin, applied identically to all T frames
        top  = np.random.randint(0, padded_size - self.image_size + 1)
        left = np.random.randint(0, padded_size - self.image_size + 1)

        return clip_padded[
            :, top : top + self.image_size, left : left + self.image_size, :
        ]  # (T, 112, 112, C)

    def _apply_centre_crop(self, clip: np.ndarray) -> np.ndarray:
        """
        Zero-pad then take the deterministic centre crop.
        Applied to VAL/TEST for consistent spatial handling.
        """
        pad = self.padding
        padded_size = self.image_size + 2 * pad

        clip_padded = np.pad(
            clip,
            pad_width=[(0, 0), (pad, pad), (pad, pad), (0, 0)],
            mode="constant",
            constant_values=0.0,
        )
        top  = (padded_size - self.image_size) // 2
        left = (padded_size - self.image_size) // 2
        return clip_padded[
            :, top : top + self.image_size, left : left + self.image_size, :
        ]

    def _compute_phase_clip_index(
        self,
        phase_frame: Optional[int],
        start_frame: int,
    ) -> int:
        """
        Map an absolute video frame index to a clip-relative index.

        A phase frame is inside the clip iff:
          start_frame <= phase_frame < start_frame + num_frames * stride
          AND (phase_frame - start_frame) % stride == 0

        Returns the clip index in [0, num_frames-1] or -1 if outside.
        """
        if phase_frame is None:
            return -1
        offset = phase_frame - start_frame
        span = self.num_frames * self.frame_stride
        if offset < 0 or offset >= span:
            return -1
        if offset % self.frame_stride != 0:
            return -1
        return offset // self.frame_stride

    def _build_phase_mask(
        self,
        filename: str,
        frame: Optional[int],
        frame_height: int,
        frame_width: int,
    ) -> torch.Tensor:
        """Rasterise the LV tracing for a given frame into a binary mask."""
        mask = torch.zeros((self.image_size, self.image_size), dtype=torch.float32)
        if frame is None or self.parser is None:
            return mask
        coords = self.parser.get_tracing_coordinates(filename, frame)
        if coords is None:
            return mask
        binary = rasterize_lv_mask(
            coords["x1"], coords["y1"],
            coords["x2"], coords["y2"],
            orig_shape=(frame_height, frame_width),
            target_shape=(self.image_size, self.image_size),
        )
        return torch.from_numpy(binary).float()

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        filename = str(row["FileName"])
        video_path = os.path.join(self.videos_dir, filename)

        # 1. Temporal sampling: choose clip start position
        start_frame = self._choose_start_frame(video_path)

        # 2. Load normalised clip  → float32 (T, H, W, C) in (x-mean)/std space
        clip = load_video_clip(
            video_path,
            start_frame=start_frame,
            num_frames=self.num_frames,
            stride=self.frame_stride,
            target_size=self.image_size,
            mean=self.mean,
            std=self.std,
        )  # (32, 112, 112, 3), float32

        # 3. Spatial augmentation (consistent crop applied to all T frames)
        if self.is_train:
            clip = self._apply_spatial_augmentation(clip)
        else:
            clip = self._apply_centre_crop(clip)

        # 4. Rearrange to (C, T, H, W) for PyTorch Conv3D
        # clip: (T, H, W, C) → (C, T, H, W)
        tensor_video = torch.from_numpy(
            np.ascontiguousarray(clip.transpose(3, 0, 1, 2))
        )  # (3, 32, 112, 112)

        # 5. LVEF regression target
        target_ef = torch.tensor(float(row["EF"]), dtype=torch.float32)

        sample: Dict[str, Any] = {
            "video": tensor_video,
            "ef": target_ef,
            "filename": filename,
        }

        # 6. Optional cardiac phase masks for XAI fidelity evaluation
        if self.return_phase_mask and self.parser is not None:
            ed_frame, es_frame = self.parser.get_cardiac_phase_frames(filename)

            frame_height = int(row.get("FrameHeight", self.image_size))
            frame_width  = int(row.get("FrameWidth",  self.image_size))

            sample["ed_mask"] = self._build_phase_mask(
                filename, ed_frame, frame_height, frame_width
            )
            sample["es_mask"] = self._build_phase_mask(
                filename, es_frame, frame_height, frame_width
            )
            # Clip-relative frame indices (-1 if outside the extracted clip)
            sample["ed_frame"] = self._compute_phase_clip_index(ed_frame, start_frame)
            sample["es_frame"] = self._compute_phase_clip_index(es_frame, start_frame)

        return sample
