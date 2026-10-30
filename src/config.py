import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetConfig:
    """EchoNet-Dynamic dataset paths and configuration."""
    # Auto-detect Kaggle vs Local environment
    is_kaggle: bool = os.path.exists("/kaggle/working")
    
    # Root directory resolution
    if is_kaggle:
        # Default Kaggle dataset locations
        root_dir: str = "/kaggle/input/echonet-dynamic/EchoNet-Dynamic"
        output_dir: str = "/kaggle/working/outputs"
    else:
        root_dir: str = "./data/EchoNet-Dynamic"
        output_dir: str = "./outputs"
        
    videos_dir: str = ""
    file_list_csv: str = ""
    volume_tracings_csv: str = ""

    def __post_init__(self):
        # Resolve full paths if root exists
        if not self.videos_dir:
            self.videos_dir = os.path.join(self.root_dir, "Videos")
        if not self.file_list_csv:
            self.file_list_csv = os.path.join(self.root_dir, "FileList.csv")
        if not self.volume_tracings_csv:
            self.volume_tracings_csv = os.path.join(self.root_dir, "VolumeTracings.csv")
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class VideoConfig:
    """Spatiotemporal video preprocessing specifications."""
    num_frames: int = 32         # Temporal window
    frame_stride: int = 1        # Temporal step size
    image_size: int = 112        # Spatial resolution (112x112)
    channels: int = 3            # Input channels (R2+1D standard)
    
    # Stanford EchoNet-Dynamic benchmark normalization constants
    mean: float = 0.126
    std: float = 0.185


@dataclass
class TrainingConfig:
    """Baseline training hyperparameters."""
    seed: int = 42
    batch_size: int = 16         # Kaggle GPU batch size (use 2 for local smoke test)
    num_workers: int = 4 if os.name != 'nt' else 0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    max_epochs: int = 30
    smooth_l1_beta: float = 1.0
    lr_min: float = 1e-6
    early_stopping_patience: int = 7
    checkpoint_dir: str = "./checkpoints"

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
