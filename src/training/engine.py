"""
EchoNet-Dynamic Training Engine
================================
Implements the full training lifecycle (mixed precision, gradient clipping,
early stopping, checkpointing) and both single-clip and multi-clip evaluation.

Multi-clip evaluation:
  During test-set evaluation each video is processed as `num_clips` evenly-spaced
  clips; the per-clip scalar predictions are averaged into a single patient-level
  LVEF estimate before metric computation.  This reduces variance from random clip
  sampling and matches the evaluation protocol used in Ouyang et al. (Nature 2020).
"""
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.metrics.clinical import compute_regression_metrics


# ---------------------------------------------------------------------------
# Single-epoch training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    max_norm: float = 1.0,
) -> float:
    """
    Execute one full training epoch with optional CUDA mixed precision (AMP).

    Gradient clipping (max_norm=1.0) is applied unconditionally to stabilise
    training on the small-EF tail of the EchoNet distribution.

    Args:
        model:      The R(2+1)D-18 regression model (in train mode).
        dataloader: Training DataLoader.
        criterion:  SmoothL1Loss (Huber).
        optimizer:  AdamW.
        device:     torch.device.
        scaler:     GradScaler for AMP (None → full fp32).
        max_norm:   Gradient clip norm.

    Returns:
        Sample-weighted average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0
    use_amp = (scaler is not None and device.type == "cuda")

    for batch in tqdm(dataloader, desc="Training", leave=False):
        videos  = batch["video"].to(device, non_blocking=True)
        targets = batch["ef"].to(device, non_blocking=True)
        bsz = videos.size(0)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                preds = model(videos)
                loss  = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(videos)
            loss  = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

        running_loss  += loss.item() * bsz
        total_samples += bsz

    return running_loss / max(1, total_samples)


# ---------------------------------------------------------------------------
# Single-clip evaluation (validation)
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate the model on a DataLoader using one centred clip per video.

    Used for validation during training where speed matters more than
    beat-to-beat averaging.

    Returns:
        (avg_loss, clinical_metrics, y_true [N,], y_pred [N,])
    """
    model.eval()
    running_loss  = 0.0
    total_samples = 0
    all_preds: List[float]  = []
    all_targets: List[float] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            videos  = batch["video"].to(device, non_blocking=True)
            targets = batch["ef"].to(device, non_blocking=True)
            bsz = videos.size(0)

            preds = model(videos)
            loss  = criterion(preds, targets)

            running_loss  += loss.item() * bsz
            total_samples += bsz

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    avg_loss = running_loss / max(1, total_samples)
    y_true   = np.asarray(all_targets, dtype=np.float64)
    y_pred   = np.asarray(all_preds,   dtype=np.float64)
    metrics  = compute_regression_metrics(y_true, y_pred)
    return avg_loss, metrics, y_true, y_pred


# ---------------------------------------------------------------------------
# Multi-clip evaluation (test set)
# ---------------------------------------------------------------------------

def evaluate_multiclip(
    model: nn.Module,
    video_paths: List[str],
    true_efs: List[float],
    device: torch.device,
    num_clips: int = 4,
    clip_len: int = 32,
    stride: int = 2,
    target_size: int = 112,
    mean: float = 0.12,
    std: float = 0.18,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Multi-clip test-time evaluation: average predictions over `num_clips`
    evenly-spaced clips per video to reduce temporal sampling variance.

    This matches the evaluation protocol described in Ouyang et al. (Nature 2020).

    Args:
        model:       Trained R(2+1)D-18 model (eval mode).
        video_paths: List of absolute paths to test videos.
        true_efs:    Corresponding ground-truth LVEF values (%).
        device:      Inference device.
        num_clips:   Number of clips to average per video (default 4).
        clip_len:    Temporal depth per clip (default 32).
        stride:      Raw-frame stride (default 2).
        target_size: Spatial size (default 112).
        mean:        Normalisation mean.
        std:         Normalisation std.

    Returns:
        (clinical_metrics, y_true [N,], y_pred [N,])
        where y_pred[i] is the average over `num_clips` clip predictions.
    """
    # Import here to avoid circular dependency at module level
    from src.data.io import load_multiclip_video

    model.eval()
    all_preds: List[float]  = []
    all_targets: List[float] = list(true_efs)

    with torch.no_grad():
        for path in tqdm(video_paths, desc="Multi-clip test eval", leave=False):
            # (num_clips, clip_len, H, W, 3)  float32 normalised
            multi_clip_np = load_multiclip_video(
                path,
                num_clips=num_clips,
                clip_len=clip_len,
                stride=stride,
                target_size=target_size,
                mean=mean,
                std=std,
            )

            # Convert to (num_clips, 3, T, H, W) tensor
            # multi_clip_np: (num_clips, T, H, W, 3) → transpose last dim first
            clips_tensor = torch.from_numpy(
                np.ascontiguousarray(
                    multi_clip_np.transpose(0, 4, 1, 2, 3)
                )
            ).float().to(device)  # (num_clips, 3, T, H, W)

            # Forward pass: batch over clips
            clip_preds = model(clips_tensor)   # (num_clips,)
            avg_pred   = float(clip_preds.mean().item())
            all_preds.append(avg_pred)

    y_true  = np.asarray(all_targets, dtype=np.float64)
    y_pred  = np.asarray(all_preds,   dtype=np.float64)
    metrics = compute_regression_metrics(y_true, y_pred)
    return metrics, y_true, y_pred


# ---------------------------------------------------------------------------
# Trainer orchestrator
# ---------------------------------------------------------------------------

class BaselineTrainer:
    """
    Orchestrates the full R(2+1)D-18 training lifecycle:
      • AMP mixed precision (CUDA only)
      • Gradient clipping (max_norm=1.0)
      • Per-epoch checkpointing on best validation MAE
      • Early stopping (patience-based)
      • JSON training history export
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        device: torch.device,
        checkpoint_dir: str = "./checkpoints",
        patience: int = 7,
        use_amp: bool = True,
    ):
        self.model         = model
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.criterion     = criterion
        self.optimizer     = optimizer
        self.scheduler     = scheduler
        self.device        = device
        self.checkpoint_dir = checkpoint_dir
        self.patience      = patience
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.scaler: Optional[torch.amp.GradScaler] = (
            torch.amp.GradScaler("cuda")
            if (use_amp and device.type == "cuda")
            else None
        )
        self.history: List[Dict[str, Any]] = []
        self.best_val_mae: float           = float("inf")
        self.best_checkpoint_path: str     = os.path.join(
            self.checkpoint_dir, "best_r2plus1d_lvef.pt"
        )

    def train(self, max_epochs: int) -> Dict[str, Any]:
        """
        Run the complete training cycle for up to `max_epochs` epochs with
        early stopping.

        Returns:
            Dict with keys: 'best_val_mae', 'best_checkpoint', 'history'.
        """
        epochs_no_improve = 0
        start_time        = time.time()

        print(
            f"\n[Trainer] Device: {self.device} | "
            f"AMP: {self.scaler is not None} | "
            f"Train batches: {len(self.train_loader)} | "
            f"Val batches: {len(self.val_loader)}"
        )

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()

            train_loss = train_one_epoch(
                self.model, self.train_loader, self.criterion,
                self.optimizer, self.device, self.scaler,
            )
            val_loss, val_metrics, _, _ = evaluate(
                self.model, self.val_loader, self.criterion, self.device
            )

            # Scheduler step
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(val_metrics["mae"])
                else:
                    self.scheduler.step()

            epoch_secs = time.time() - epoch_start
            val_mae    = val_metrics["mae"]
            val_rmse   = val_metrics["rmse"]
            val_r      = val_metrics["pearson_r"]

            record: Dict[str, Any] = {
                "epoch":        epoch,
                "train_loss":   train_loss,
                "val_loss":     val_loss,
                "val_mae":      val_mae,
                "val_rmse":     val_rmse,
                "val_pearson_r": val_r,
                "lr":           current_lr,
                "duration_seconds": epoch_secs,
            }
            self.history.append(record)

            print(
                f"Epoch [{epoch:02d}/{max_epochs:02d}]  "
                f"Train Loss: {train_loss:.4f}  "
                f"Val Loss: {val_loss:.4f}  "
                f"Val MAE: {val_mae:.2f}%  "
                f"Val RMSE: {val_rmse:.2f}%  "
                f"Pearson r: {val_r:.3f}  "
                f"LR: {current_lr:.2e}  "
                f"Time: {epoch_secs:.1f}s"
            )

            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                epochs_no_improve = 0
                payload = {
                    "epoch":              epoch,
                    "model_state_dict":   self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_val_mae":       self.best_val_mae,
                    "val_metrics":        val_metrics,
                }
                torch.save(payload, self.best_checkpoint_path)
                print(
                    f"  → Best checkpoint saved: {self.best_checkpoint_path} "
                    f"(MAE: {val_mae:.2f}%)"
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(
                        f"\n[Trainer] Early stopping triggered at epoch {epoch} "
                        f"({self.patience} epochs without improvement)."
                    )
                    break

        total_min = (time.time() - start_time) / 60.0
        print(
            f"\n[Trainer] Training complete in {total_min:.2f} min | "
            f"Best Val MAE: {self.best_val_mae:.2f}%"
        )

        history_path = os.path.join(self.checkpoint_dir, "training_history.json")
        with open(history_path, "w") as fh:
            json.dump(self.history, fh, indent=2)

        return {
            "best_val_mae":    self.best_val_mae,
            "best_checkpoint": self.best_checkpoint_path,
            "history":         self.history,
        }
