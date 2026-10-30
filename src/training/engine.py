import os
import json
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics.clinical import compute_regression_metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    max_norm: float = 1.0
) -> float:
    """
    Execute one training epoch with optional mixed precision.
    
    Returns:
        Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0
    use_amp = (scaler is not None and device.type == "cuda")

    for batch in tqdm(dataloader, desc="Training", leave=False):
        videos = batch["video"].to(device, non_blocking=True)
        targets = batch["ef"].to(device, non_blocking=True)
        batch_size = videos.size(0)

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

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(1, total_samples)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate model on validation or test dataset.
    
    Returns:
        (avg_loss, clinical_metrics_dict, y_true_array, y_pred_array)
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            videos = batch["video"].to(device, non_blocking=True)
            targets = batch["ef"].to(device, non_blocking=True)
            batch_size = videos.size(0)

            preds = model(videos)
            loss = criterion(preds, targets)

            running_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    avg_loss = running_loss / max(1, total_samples)
    y_true = np.array(all_targets, dtype=np.float64)
    y_pred = np.array(all_preds, dtype=np.float64)

    metrics = compute_regression_metrics(y_true, y_pred)
    return avg_loss, metrics, y_true, y_pred


class BaselineTrainer:
    """
    Trainer orchestrating full training lifecycle, early stopping,
    checkpointing, and performance logging.
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
        use_amp: bool = True
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

        self.scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None
        self.history: List[Dict[str, Any]] = []
        self.best_val_mae = float("inf")
        self.best_checkpoint_path = os.path.join(self.checkpoint_dir, "best_r2plus1d_lvef.pt")

    def train(self, max_epochs: int) -> Dict[str, Any]:
        """Run complete training cycle across epochs."""
        epochs_no_improve = 0
        start_time = time.time()

        print(f"\n[Trainer] Starting training on device: {self.device} (AMP: {self.scaler is not None})")
        print(f"[Trainer] Train batches: {len(self.train_loader)} | Val batches: {len(self.val_loader)}")

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()

            train_loss = train_one_epoch(
                self.model, self.train_loader, self.criterion,
                self.optimizer, self.device, self.scaler
            )

            val_loss, val_metrics, _, _ = evaluate(
                self.model, self.val_loader, self.criterion, self.device
            )

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["mae"])
                else:
                    self.scheduler.step()

            epoch_duration = time.time() - epoch_start
            val_mae = val_metrics["mae"]
            val_rmse = val_metrics["rmse"]
            val_r = val_metrics["pearson_r"]

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "val_pearson_r": val_r,
                "lr": current_lr,
                "duration_seconds": epoch_duration
            }
            self.history.append(epoch_record)

            print(
                f"Epoch [{epoch:02d}/{max_epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val MAE: {val_mae:.2f}% | "
                f"Val RMSE: {val_rmse:.2f}% | "
                f"Pearson r: {val_r:.3f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_duration:.1f}s"
            )

            # Checkpoint on best validation MAE
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                epochs_no_improve = 0
                checkpoint_payload = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_val_mae": self.best_val_mae,
                    "val_metrics": val_metrics
                }
                torch.save(checkpoint_payload, self.best_checkpoint_path)
                print(f" -> Best checkpoint saved to: {self.best_checkpoint_path} (MAE: {val_mae:.2f}%)")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"\n[Trainer] Early stopping triggered after {epoch} epochs.")
                    break

        total_elapsed = time.time() - start_time
        print(f"\n[Trainer] Training complete in {total_elapsed / 60:.2f} minutes.")
        print(f"[Trainer] Best Validation MAE: {self.best_val_mae:.2f}%")

        # Save history log
        history_path = os.path.join(self.checkpoint_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return {
            "best_val_mae": self.best_val_mae,
            "best_checkpoint": self.best_checkpoint_path,
            "history": self.history
        }
