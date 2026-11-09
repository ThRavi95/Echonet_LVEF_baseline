"""
EchoNet-Dynamic LVEF Inference Pipeline
=========================================
Provides deterministic multi-clip inference from a trained R(2+1)D-18 checkpoint.
No synthetic fallbacks.  A FileNotFoundError is raised if the checkpoint is absent.
"""
import os
from typing import Any, Dict

import numpy as np
import torch

from src.config import VideoConfig
from src.data.io import load_multiclip_video
from src.models.r2plus1d import EchoNetR2Plus1D


class LVEFEstimator:
    """
    Inference wrapper for the trained R(2+1)D-18 LVEF regression model.

    Uses multi-clip averaging (4 evenly-spaced clips) for robust prediction.

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
        device: str = None,
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

        self.model = EchoNetR2Plus1D(pretrained=False, dropout_rate=0.0)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        print(f"[LVEFEstimator] Checkpoint loaded: {checkpoint_path}")

    @torch.no_grad()
    def predict(self, video_path: str) -> Dict[str, Any]:
        """
        Predict LVEF from a single video file using multi-clip averaging.

        Args:
            video_path: Path to an .avi echocardiogram (or .npy companion).

        Returns:
            Dict with keys: predicted_lvef, raw_output, clinical_category,
            badge_color, ase_classification, summary.
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

        # → (num_clips, 3, T, H, W)
        clips_tensor = torch.from_numpy(
            np.ascontiguousarray(multi_clip.transpose(0, 4, 1, 2, 3))
        ).float().to(self.device)

        preds = self.model(clips_tensor)        # (num_clips,)
        raw_pred = float(preds.mean().item())
        # Clamp to physiologically plausible range
        clamped_ef = float(np.clip(raw_pred, 10.0, 85.0))

        report = self._clinical_report(clamped_ef)
        return {
            "predicted_lvef":     round(clamped_ef, 1),
            "raw_output":         round(raw_pred, 2),
            "clinical_category":  report["category"],
            "badge_color":        report["badge_color"],
            "ase_classification": report["ase_classification"],
            "summary":            report["summary"],
        }

    @staticmethod
    def _clinical_report(ef: float) -> Dict[str, str]:
        """Translate LVEF into ASE guideline-based clinical classification."""
        if ef >= 52.0:
            return {
                "category": "Normal / Preserved EF (HFpEF)",
                "badge_color": "#10b981",
                "ase_classification": "Normal LV Systolic Function",
                "summary": (
                    f"Estimated LVEF {ef:.1f}% — normal global LV systolic function. "
                    "Myocardial contractility and endocardial excursion are preserved "
                    "across the full cardiac cycle."
                ),
            }
        elif ef >= 41.0:
            return {
                "category": "Mildly Reduced EF (HFmrEF)",
                "badge_color": "#f59e0b",
                "ase_classification": "Mild LV Systolic Dysfunction",
                "summary": (
                    f"Estimated LVEF {ef:.1f}% — mildly reduced systolic capacity. "
                    "Moderate attenuation of apical and basal endocardial excursion "
                    "during systole. Clinical correlation recommended."
                ),
            }
        elif ef >= 30.0:
            return {
                "category": "Moderately Reduced EF",
                "badge_color": "#f97316",
                "ase_classification": "Moderate LV Systolic Dysfunction",
                "summary": (
                    f"Estimated LVEF {ef:.1f}% — moderate systolic impairment. "
                    "Noticeable hypokinesis and compromised stroke volume. "
                    "Cardiology evaluation and guideline-directed therapy advised."
                ),
            }
        else:
            return {
                "category": "Severely Reduced EF (HFrEF)",
                "badge_color": "#ef4444",
                "ase_classification": "Severe LV Systolic Dysfunction",
                "summary": (
                    f"Estimated LVEF {ef:.1f}% — severe systolic dysfunction. "
                    "Diffuse myocardial hypokinesis and markedly reduced fractional "
                    "shortening. Urgent clinical correlation required."
                ),
            }
