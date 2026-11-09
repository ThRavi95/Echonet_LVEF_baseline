"""
EchoNet-Dynamic R(2+1)D-18 Baseline — Configuration
=====================================================
All constants are aligned with Ouyang et al., Nature 2020:
  - num_frames=32, frame_stride=2 → 64-frame window ≈ 1.28s at 50 fps,
    capturing at least one full cardiac cycle.
  - Spatial resolution: 112×112 (R(2+1)D-18 standard).
  - Normalization: frames are first cast to float32 and scaled to [0.0, 1.0],
    then standardised via (x - mean) / std using the EchoNet-Dynamic dataset
    statistics (mean≈0.12, std≈0.18 in the [0,1] domain).
"""
import os
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """EchoNet-Dynamic dataset paths and environment-aware root resolution."""

    # Auto-detect Kaggle vs local environment
    is_kaggle: bool = field(default_factory=lambda: os.path.exists("/kaggle/working"))

    # Paths are resolved in __post_init__ to allow is_kaggle to be evaluated first
    root_dir: str = ""
    output_dir: str = ""
    videos_dir: str = ""
    file_list_csv: str = ""
    volume_tracings_csv: str = ""

    def __post_init__(self):
        if not self.root_dir:
            self.root_dir = (
                "/kaggle/input/echonet-dynamic/EchoNet-Dynamic"
                if self.is_kaggle
                else "./data/EchoNet-Dynamic"
            )
        if not self.output_dir:
            self.output_dir = (
                "/kaggle/working/outputs" if self.is_kaggle else "./outputs"
            )
        if not self.videos_dir:
            self.videos_dir = os.path.join(self.root_dir, "Videos")
        if not self.file_list_csv:
            self.file_list_csv = os.path.join(self.root_dir, "FileList.csv")
        if not self.volume_tracings_csv:
            self.volume_tracings_csv = os.path.join(self.root_dir, "VolumeTracings.csv")
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class VideoConfig:
    """
    Spatiotemporal video preprocessing specification.

    Normalization pipeline (must be applied in this exact order):
      1. Load frame as uint8 [0, 255].
      2. Cast to float32 and divide by 255.0  →  [0.0, 1.0].
      3. Standardise: (x - mean) / std        →  zero-centred, unit-ish variance.

    EchoNet-Dynamic channel statistics (computed over the training set, [0,1] domain):
      mean = 0.12,  std = 0.18
    """

    # ----- Temporal -------------------------------------------------------
    num_frames: int = 32
    # Stride-2 sampling: every other raw frame is selected.
    # Effective window = num_frames * frame_stride = 64 raw frames ≈ 1.28 s @ 50 fps.
    frame_stride: int = 2

    # ----- Spatial --------------------------------------------------------
    image_size: int = 112          # Final H=W after crop (R(2+1)D-18 canonical input)
    channels: int = 3              # RGB input channels

    # Spatial augmentation: zero-pad each spatial side by `padding` pixels (→ 136×136),
    # then random-crop back to image_size×image_size (train only; val uses centre crop).
    padding: int = 12              # 112 + 2*12 = 136 → random crop to 112

    # ----- Normalisation --------------------------------------------------
    # Applied AFTER scaling uint8 → float32 / 255.0.
    mean: float = 0.12
    std: float = 0.18


@dataclass
class TrainingConfig:
    """Baseline training hyperparameters — faithful to Ouyang et al. (Nature 2020)."""

    seed: int = 42
    batch_size: int = 16          # 16 GB GPU; use 2 for local CPU smoke tests
    num_workers: int = 4 if os.name != "nt" else 0
    pin_memory: bool = True

    # Optimiser
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2    # AdamW L2 regularisation

    # Scheduler
    max_epochs: int = 30
    lr_min: float = 1e-6          # CosineAnnealingLR η_min

    # Loss
    smooth_l1_beta: float = 1.0   # Huber / Smooth-L1 delta

    # Early stopping
    early_stopping_patience: int = 7

    # Checkpoint
    checkpoint_dir: str = "./checkpoints"

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
