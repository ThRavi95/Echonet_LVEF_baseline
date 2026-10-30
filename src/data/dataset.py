import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

from src.data.io import load_video, sample_video_clip
from src.data.rasterizer import rasterize_lv_mask
from src.data.parser import EchoNetParser


class EchoNetVideoDataset(Dataset):
    """
    PyTorch Dataset for EchoNet-Dynamic video-based LVEF regression.
    Produces 5D tensors of shape: (Channels=3, Frames=32, Height=112, Width=112).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        videos_dir: str,
        num_frames: int = 32,
        frame_stride: int = 1,
        image_size: int = 112,
        is_train: bool = True,
        mean: float = 0.126,
        std: float = 0.185,
        parser: Optional[EchoNetParser] = None,
        return_phase_mask: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.videos_dir = videos_dir
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.image_size = image_size
        self.is_train = is_train
        self.mean = mean
        self.std = std
        self.parser = parser
        self.return_phase_mask = return_phase_mask

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        filename = str(row["FileName"])
        video_path = os.path.join(self.videos_dir, filename)

        # 1. Load full video
        # Shape: (T_orig, H, W, 3)
        raw_frames = load_video(video_path, target_size=self.image_size)
        total_orig_frames = len(raw_frames)

        # 2. Phase-aware clip sampling
        span = self.num_frames * self.frame_stride
        start_frame = None

        if self.is_train:
            # Random clip window for data diversity
            if total_orig_frames > span:
                start_frame = np.random.randint(0, total_orig_frames - span)
        else:
            # Deterministic centered clip window for validation/testing
            start_frame = max(0, (total_orig_frames - span) // 2)

        clip_frames = sample_video_clip(
            raw_frames,
            target_frames=self.num_frames,
            stride=self.frame_stride,
            start_frame=start_frame
        )

        # 3. Normalization: convert to float in [0, 1] then standardize
        # clip_frames shape: (32, 112, 112, 3) -> transpose to (3, 32, 112, 112)
        tensor_video = torch.from_numpy(clip_frames).float() / 255.0
        tensor_video = (tensor_video - self.mean) / self.std
        tensor_video = tensor_video.permute(3, 0, 1, 2)  # (C, T, H, W)

        # 4. Target LVEF scalar
        target_ef = torch.tensor(float(row["EF"]), dtype=torch.float32)

        sample = {
            "video": tensor_video,
            "ef": target_ef,
            "filename": filename
        }

        # 5. Optional Ground-Truth LV masks for XAI Fidelity
        if self.return_phase_mask and self.parser is not None:
            ed_frame, es_frame = self.parser.get_cardiac_phase_frames(filename)
            ed_mask = torch.zeros((self.image_size, self.image_size), dtype=torch.float32)
            es_mask = torch.zeros((self.image_size, self.image_size), dtype=torch.float32)

            if ed_frame is not None:
                ed_coords = self.parser.get_tracing_coordinates(filename, ed_frame)
                if ed_coords:
                    m = rasterize_lv_mask(
                        ed_coords["x1"], ed_coords["y1"],
                        ed_coords["x2"], ed_coords["y2"],
                        orig_shape=(row.get("FrameHeight", 112), row.get("FrameWidth", 112)),
                        target_shape=(self.image_size, self.image_size)
                    )
                    ed_mask = torch.from_numpy(m).float()

            if es_frame is not None:
                es_coords = self.parser.get_tracing_coordinates(filename, es_frame)
                if es_coords:
                    m = rasterize_lv_mask(
                        es_coords["x1"], es_coords["y1"],
                        es_coords["x2"], es_coords["y2"],
                        orig_shape=(row.get("FrameHeight", 112), row.get("FrameWidth", 112)),
                        target_shape=(self.image_size, self.image_size)
                    )
                    es_mask = torch.from_numpy(m).float()

            sample["ed_mask"] = ed_mask
            sample["es_mask"] = es_mask
            sample["ed_frame"] = ed_frame if ed_frame is not None else -1
            sample["es_frame"] = es_frame if es_frame is not None else -1

        return sample
