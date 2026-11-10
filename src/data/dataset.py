"""
EchoNet-Dynamic PyTorch Datasets
=================================
Implements strict split-aware sampling, spatial augmentation, normalisation,
and optional cardiac phase mask generation aligned with Ouyang et al. (Nature 2020).
Defensively validates all video files on disk (handling case differences like
.avi vs .AVI) and excludes missing/corrupted entries before DataLoader execution.
"""
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.io import (
    get_video_total_frames,
    load_multiclip_video,
    load_video_clip,
)
from src.data.parser import EchoNetParser
from src.data.rasterizer import rasterize_lv_mask


class EchoNetVideoDataset(Dataset):
    """
    PyTorch Dataset for EchoNet-Dynamic video-based LVEF regression.

    Output tensor shape: (3, num_frames, image_size, image_size) — (C, T, H, W).
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

        # Index actual disk directory case-insensitively
        self._build_disk_index()

        # Defensively resolve video paths (.avi, .AVI, companion .npy) and drop missing files
        initial_count = len(self.df)
        self.df["resolved_video_path"] = self.df["FileName"].apply(self._resolve_path)
        self.df = self.df[self.df["resolved_video_path"] != ""].reset_index(drop=True)
        dropped_count = initial_count - len(self.df)
        if dropped_count > 0:
            print(
                f"[Dataset] Verified {len(self.df)}/{initial_count} videos on disk "
                f"({dropped_count} missing or empty files dropped from split)."
            )

    def _build_disk_index(self) -> None:
        """Scan videos_dir once and build case-insensitive name -> absolute path map."""
        self.disk_files: Dict[str, str] = {}
        if os.path.exists(self.videos_dir):
            try:
                for entry in os.scandir(self.videos_dir):
                    if entry.is_file():
                        self.disk_files[entry.name.lower()] = entry.path
            except Exception:
                pass

    def _resolve_path(self, filename: str) -> str:
        s = str(filename).strip()
        base, _ = os.path.splitext(s)
        base_clean = base.strip()

        candidates = [
            s.lower(),
            f"{base_clean}.avi".lower(),
            f"{base_clean}.npy".lower(),
            f"{base_clean}.mp4".lower(),
            f"{base_clean}.avi",
            f"{base_clean}.AVI",
        ]
        for c in candidates:
            c_lower = c.lower()
            if c_lower in self.disk_files:
                path = self.disk_files[c_lower]
                try:
                    if os.path.getsize(path) > 0:
                        return path
                except OSError:
                    pass

        # Fallback to direct path checks (e.g. symlinks or relative paths)
        for c_name in [s, f"{base_clean}.avi", f"{base_clean}.AVI", f"{base_clean}.npy"]:
            direct = os.path.join(self.videos_dir, c_name)
            if os.path.exists(direct):
                try:
                    if os.path.getsize(direct) > 0:
                        return direct
                except OSError:
                    pass

        return ""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _choose_start_frame(self, video_path: str) -> int:
        total = get_video_total_frames(video_path)
        span = self.num_frames * self.frame_stride

        if total <= span:
            return 0

        max_start = total - span
        if self.is_train:
            return int(np.random.randint(0, max_start + 1))
        return int(max_start // 2)

    def _apply_spatial_augmentation(
        self, clip: np.ndarray
    ) -> Tuple[np.ndarray, int, int]:
        pad = self.padding
        if pad <= 0:
            return clip, 0, 0

        padded_size = self.image_size + 2 * pad
        clip_padded = np.pad(
            clip,
            pad_width=[(0, 0), (pad, pad), (pad, pad), (0, 0)],
            mode="constant",
            constant_values=0.0,
        )

        top = int(np.random.randint(0, padded_size - self.image_size + 1))
        left = int(np.random.randint(0, padded_size - self.image_size + 1))
        cropped = clip_padded[
            :, top : top + self.image_size, left : left + self.image_size, :
        ]
        return cropped, top, left

    def _crop_mask_to_match(
        self, mask: np.ndarray, top: int, left: int
    ) -> np.ndarray:
        pad = self.padding
        if pad <= 0:
            return mask
        mask_padded = np.pad(
            mask,
            pad_width=[(pad, pad), (pad, pad)],
            mode="constant",
            constant_values=0,
        )
        return mask_padded[
            top : top + self.image_size, left : left + self.image_size
        ]

    def _compute_phase_clip_index(
        self,
        phase_frame: Optional[int],
        start_frame: int,
    ) -> int:
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
        crop_top: Optional[int] = None,
        crop_left: Optional[int] = None,
    ) -> torch.Tensor:
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        if frame is not None and self.parser is not None:
            coords = self.parser.get_tracing_coordinates(filename, frame)
            if coords is not None:
                mask = rasterize_lv_mask(
                    coords["x1"],
                    coords["y1"],
                    coords["x2"],
                    coords["y2"],
                    orig_shape=(frame_height, frame_width),
                    target_shape=(self.image_size, self.image_size),
                )

        if crop_top is not None and crop_left is not None:
            mask = self._crop_mask_to_match(mask, crop_top, crop_left)

        return torch.from_numpy(np.ascontiguousarray(mask)).float()

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        filename = str(row["FileName"])
        video_path = str(row["resolved_video_path"])

        try:
            # 1. Temporal sampling
            start_frame = self._choose_start_frame(video_path)

            # 2. Load normalised clip -> float32 (T, H, W, C)
            clip = load_video_clip(
                video_path,
                start_frame=start_frame,
                num_frames=self.num_frames,
                stride=self.frame_stride,
                target_size=self.image_size,
                mean=self.mean,
                std=self.std,
            )
        except Exception as exc:
            # Fallback to next sample if an individual file is unreadable
            if len(self.df) > 1:
                return self.__getitem__((idx + 1) % len(self.df))
            raise exc

        # 3. Spatial augmentation (TRAIN only; VAL/TEST use native 112x112)
        crop_top: Optional[int] = None
        crop_left: Optional[int] = None
        if self.is_train:
            clip, crop_top, crop_left = self._apply_spatial_augmentation(clip)

        # 4. Rearrange to (C, T, H, W)
        tensor_video = torch.from_numpy(
            np.ascontiguousarray(clip.transpose(3, 0, 1, 2))
        )

        # 5. LVEF regression target
        target_ef = torch.tensor(float(row["EF"]), dtype=torch.float32)

        sample: Dict[str, Any] = {
            "video": tensor_video,
            "ef": target_ef,
            "filename": filename,
        }

        # 6. Optional cardiac phase masks
        if self.return_phase_mask and self.parser is not None:
            ed_frame, es_frame = self.parser.get_cardiac_phase_frames(filename)
            frame_height = int(row.get("FrameHeight", self.image_size))
            frame_width = int(row.get("FrameWidth", self.image_size))

            sample["ed_mask"] = self._build_phase_mask(
                filename, ed_frame, frame_height, frame_width, crop_top, crop_left
            )
            sample["es_mask"] = self._build_phase_mask(
                filename, es_frame, frame_height, frame_width, crop_top, crop_left
            )
            sample["ed_frame"] = self._compute_phase_clip_index(ed_frame, start_frame)
            sample["es_frame"] = self._compute_phase_clip_index(es_frame, start_frame)

        return sample


class EchoNetMultiClipDataset(Dataset):
    """
    Multi-clip test-time Dataset for DataLoader-parallelised evaluation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        videos_dir: str,
        num_clips: int = 4,
        num_frames: int = 32,
        frame_stride: int = 2,
        image_size: int = 112,
        mean: float = 0.12,
        std: float = 0.18,
    ):
        self.df = df.reset_index(drop=True)
        self.videos_dir = videos_dir
        self.num_clips = num_clips
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.image_size = image_size
        self.mean = mean
        self.std = std

        self._build_disk_index()
        initial_count = len(self.df)
        self.df["resolved_video_path"] = self.df["FileName"].apply(self._resolve_path)
        self.df = self.df[self.df["resolved_video_path"] != ""].reset_index(drop=True)
        dropped_count = initial_count - len(self.df)
        if dropped_count > 0:
            print(
                f"[TestDataset] Verified {len(self.df)}/{initial_count} videos on disk "
                f"({dropped_count} missing or empty files dropped from test split)."
            )

    def _build_disk_index(self) -> None:
        self.disk_files: Dict[str, str] = {}
        if os.path.exists(self.videos_dir):
            try:
                for entry in os.scandir(self.videos_dir):
                    if entry.is_file():
                        self.disk_files[entry.name.lower()] = entry.path
            except Exception:
                pass

    def _resolve_path(self, filename: str) -> str:
        s = str(filename).strip()
        base, _ = os.path.splitext(s)
        base_clean = base.strip()

        candidates = [
            s.lower(),
            f"{base_clean}.avi".lower(),
            f"{base_clean}.npy".lower(),
            f"{base_clean}.mp4".lower(),
            f"{base_clean}.avi",
            f"{base_clean}.AVI",
        ]
        for c in candidates:
            c_lower = c.lower()
            if c_lower in self.disk_files:
                path = self.disk_files[c_lower]
                try:
                    if os.path.getsize(path) > 0:
                        return path
                except OSError:
                    pass

        for c_name in [s, f"{base_clean}.avi", f"{base_clean}.AVI", f"{base_clean}.npy"]:
            direct = os.path.join(self.videos_dir, c_name)
            if os.path.exists(direct):
                try:
                    if os.path.getsize(direct) > 0:
                        return direct
                except OSError:
                    pass

        return ""

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        filename = str(row["FileName"])
        video_path = str(row["resolved_video_path"])

        try:
            multi_clip_np = load_multiclip_video(
                video_path,
                num_clips=self.num_clips,
                clip_len=self.num_frames,
                stride=self.frame_stride,
                target_size=self.image_size,
                mean=self.mean,
                std=self.std,
            )
        except Exception as exc:
            if len(self.df) > 1:
                return self.__getitem__((idx + 1) % len(self.df))
            raise exc

        clips_tensor = torch.from_numpy(
            np.ascontiguousarray(multi_clip_np.transpose(0, 4, 1, 2, 3))
        ).float()

        return {
            "clips": clips_tensor,
            "ef": torch.tensor(float(row["EF"]), dtype=torch.float32),
            "filename": filename,
        }
