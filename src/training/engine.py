"""
EchoNet-Dynamic Training Engine
================================
Implements the full training lifecycle (mixed precision, gradient clipping,
early stopping, fault-tolerant checkpointing & resumption) and both
single-clip and DataLoader-parallelised multi-clip evaluation.

Multi-clip evaluation:
  During test-set evaluation each video is processed as `num_clips` evenly-spaced
  clips via EchoNetMultiClipDataset + DataLoader; the per-clip scalar predictions
  are averaged into a single patient-level LVEF estimate before metric computation.
"""
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

    Returns:
        Sample-weighted average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0
    use_amp = (scaler is not None and device.type == "cuda")

    for batch in tqdm(dataloader, desc="Training", leave=False):
        videos = batch["video"].to(device, non_blocking=True)
        targets = batch["ef"].to(device, non_blocking=True)
        bsz = videos.size(0)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                preds = model(videos)
                loss = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(videos)
            loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

        running_loss += loss.item() * bsz
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

    Returns:
        (avg_loss, clinical_metrics, y_true [N,], y_pred [N,])
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds: List[float] = []
    all_targets: List[float] = []
    use_amp = (device.type == "cuda")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            videos = batch["video"].to(device, non_blocking=True)
            targets = batch["ef"].to(device, non_blocking=True)
            bsz = videos.size(0)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    preds = model(videos)
                    loss = criterion(preds, targets)
            else:
                preds = model(videos)
                loss = criterion(preds, targets)

            running_loss += loss.item() * bsz
            total_samples += bsz

            all_preds.extend(preds.float().cpu().tolist())
            all_targets.extend(targets.float().cpu().tolist())

    avg_loss = running_loss / max(1, total_samples)
    y_true = np.asarray(all_targets, dtype=np.float64)
    y_pred = np.asarray(all_preds, dtype=np.float64)
    metrics = compute_regression_metrics(y_true, y_pred)
    return avg_loss, metrics, y_true, y_pred


# ---------------------------------------------------------------------------
# Multi-clip evaluation (DataLoader-parallelised test set)
# ---------------------------------------------------------------------------

def evaluate_multiclip(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Multi-clip test-time evaluation using a DataLoader over EchoNetMultiClipDataset.

    Each batch provides:
      clips: (B, K, 3, T, H, W)
      ef:    (B,)

    All B*K clips are inferred in parallel and averaged per patient/video.
    """
    model.eval()
    all_preds: List[float] = []
    all_targets: List[float] = []
    use_amp = (device.type == "cuda")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Multi-clip test eval", leave=False):
            clips = batch["clips"].to(device, non_blocking=True)  # (B, K, C, T, H, W)
            targets = batch["ef"]
            bsz, num_clips, ch, frames, h, w = clips.shape
            flat_clips = clips.view(bsz * num_clips, ch, frames, h, w)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    flat_preds = model(flat_clips)
            else:
                flat_preds = model(flat_clips)

            patient_preds = flat_preds.float().view(bsz, num_clips).mean(dim=1)
            all_preds.extend(patient_preds.cpu().tolist())
            all_targets.extend(targets.float().cpu().tolist())

    y_true = np.asarray(all_targets, dtype=np.float64)
    y_pred = np.asarray(all_preds, dtype=np.float64)
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
      • Per-epoch checkpointing: saves both `last_checkpoint.pt` (for session
        resumption) and `best_r2plus1d_lvef.pt` (best validation MAE)
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
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.patience = patience
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.scaler: Optional[torch.amp.GradScaler] = (
            torch.amp.GradScaler("cuda")
            if (use_amp and device.type == "cuda")
            else None
        )
        self.history: List[Dict[str, Any]] = []
        self.best_val_mae: float = float("inf")
        self.best_checkpoint_path: str = os.path.join(
            self.checkpoint_dir, "best_r2plus1d_lvef.pt"
        )
        self.last_checkpoint_path: str = os.path.join(
            self.checkpoint_dir, "last_checkpoint.pt"
        )

    def _save_last_checkpoint(self, epoch: int, epochs_no_improve: int) -> None:
        payload: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler_state_dict": (
                self.scaler.state_dict() if self.scaler is not None else None
            ),
            "best_val_mae": self.best_val_mae,
            "epochs_no_improve": epochs_no_improve,
            "history": self.history,
        }
        torch.save(payload, self.last_checkpoint_path)

    def _load_resume_checkpoint(self, resume_path: str) -> Tuple[int, int]:
        """Restore trainer state from checkpoint and return (start_epoch, epochs_no_improve)."""
        ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if (
            self.scheduler is not None
            and ckpt.get("scheduler_state_dict") is not None
        ):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if self.scaler is not None and ckpt.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        self.best_val_mae = float(ckpt.get("best_val_mae", float("inf")))
        self.history = list(ckpt.get("history", []))
        last_epoch = int(ckpt.get("epoch", 0))
        epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
        print(
            f"[Trainer] Resumed from {resume_path} at epoch {last_epoch} "
            f"(best Val MAE: {self.best_val_mae:.2f}%)"
        )
        return last_epoch + 1, epochs_no_improve

    def train(
        self,
        max_epochs: int,
        resume: bool = False,
        resume_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete training cycle for up to `max_epochs` epochs with
        early stopping and fault-tolerant resumption.
        """
        start_epoch = 1
        epochs_no_improve = 0
        ckpt_to_resume = resume_path or self.last_checkpoint_path
        if resume and os.path.exists(ckpt_to_resume):
            start_epoch, epochs_no_improve = self._load_resume_checkpoint(
                ckpt_to_resume
            )
        elif resume:
            print(
                f"[Trainer] --resume requested but no checkpoint found at "
                f"{ckpt_to_resume}; starting from epoch 1."
            )

        start_time = time.time()

        print(
            f"\n[Trainer] Device: {self.device} | "
            f"AMP: {self.scaler is not None} | "
            f"Epochs: {start_epoch}..{max_epochs} | "
            f"Train batches: {len(self.train_loader)} | "
            f"Val batches: {len(self.val_loader)}"
        )

        for epoch in range(start_epoch, max_epochs + 1):
            epoch_start = time.time()

            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                self.scaler,
            )
            val_loss, val_metrics, _, _ = evaluate(
                self.model, self.val_loader, self.criterion, self.device
            )

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
            val_mae = val_metrics["mae"]
            val_rmse = val_metrics["rmse"]
            val_r = val_metrics["pearson_r"]

            record: Dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "val_pearson_r": val_r,
                "lr": current_lr,
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
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_val_mae": self.best_val_mae,
                    "val_metrics": val_metrics,
                }
                torch.save(payload, self.best_checkpoint_path)
                print(
                    f"  → Best checkpoint saved: {self.best_checkpoint_path} "
                    f"(MAE: {val_mae:.2f}%)"
                )
            else:
                epochs_no_improve += 1

            # Always save resumable last_checkpoint.pt at the end of every epoch
            self._save_last_checkpoint(epoch, epochs_no_improve)

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
            "best_val_mae": self.best_val_mae,
            "best_checkpoint": self.best_checkpoint_path,
            "last_checkpoint": self.last_checkpoint_path,
            "history": self.history,
        }
