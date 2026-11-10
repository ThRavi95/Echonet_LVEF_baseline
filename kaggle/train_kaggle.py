"""
EchoNet-Dynamic R(2+1)D-18 Baseline — Kaggle GPU Training Script
=================================================================
Trains the R(2+1)D-18 regression model on the full EchoNet-Dynamic dataset
(7,465 training videos) and evaluates on the official 1,277-video test set
using DataLoader-parallelised multi-clip averaging.

Usage (Kaggle notebook cell):
    !python kaggle/train_kaggle.py \\
        --data_dir /kaggle/input/echonet-dynamic/EchoNet-Dynamic \\
        --output_dir /kaggle/working/outputs \\
        --epochs 25 \\
        --batch_size 16 \\
        --resume

Dataset attachment (Kaggle):
    1. Click '+ Add Input' in the right sidebar.
    2. Search for 'echonet-dynamic'.
    3. Add the dataset containing FileList.csv, VolumeTracings.csv, and Videos/.

This script will raise FileNotFoundError immediately if the dataset is not
found — no synthetic fallbacks exist.
"""
import argparse
import json
import os
import sys
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure repository root is importable from Kaggle working directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import TrainingConfig, VideoConfig
from src.data.dataset import EchoNetMultiClipDataset, EchoNetVideoDataset
from src.data.parser import EchoNetParser
from src.metrics.clinical import plot_bland_altman, plot_scatter_correlation
from src.models.r2plus1d import EchoNetR2Plus1D
from src.training.engine import BaselineTrainer, evaluate_multiclip


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def find_dataset_components(hint_path: str = "") -> Tuple[str, Optional[str], str]:
    """
    Locate FileList.csv, VolumeTracings.csv, and the Videos/ folder.

    Search order:
      1. hint_path (--data_dir argument), if provided.
      2. /kaggle/input (recursive scan for Kaggle attached datasets).
      3. ./data/EchoNet-Dynamic and ./data (local fallback).
    """
    search_roots = []
    if hint_path:
        search_roots.append(hint_path)
    search_roots.extend([
        "/kaggle/input",
        "./data/EchoNet-Dynamic",
        "./data",
    ])

    file_list_path: Optional[str] = None
    vol_tracings_path: Optional[str] = None
    videos_dir: Optional[str] = None

    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            for f in filenames:
                fl = f.lower()
                if fl == "filelist.csv" and file_list_path is None:
                    file_list_path = os.path.join(dirpath, f)
                elif fl == "volumetracings.csv" and vol_tracings_path is None:
                    vol_tracings_path = os.path.join(dirpath, f)
            for d in dirnames:
                if d.lower() in ("videos", "video") and videos_dir is None:
                    videos_dir = os.path.join(dirpath, d)

    if file_list_path is None:
        kaggle_contents = ""
        if os.path.exists("/kaggle/input"):
            entries = os.listdir("/kaggle/input")
            kaggle_contents = f"\nContents of /kaggle/input: {entries}"
            for item in entries:
                sub = os.path.join("/kaggle/input", item)
                if os.path.isdir(sub):
                    kaggle_contents += f"\n  └─ {item}/: {os.listdir(sub)[:10]}"

        raise FileNotFoundError(
            "\n" + "=" * 65 + "\n"
            "ERROR: EchoNet-Dynamic dataset not found.\n\n"
            "To attach the dataset on Kaggle:\n"
            "  1. Click '+ Add Input' in the right sidebar.\n"
            "  2. Search for 'echonet dynamic'.\n"
            "  3. Add the dataset containing FileList.csv, VolumeTracings.csv,\n"
            "     and a Videos/ folder.\n"
            "  4. Re-run this cell.\n\n"
            "To run locally:\n"
            "  python kaggle/train_kaggle.py \\\n"
            "      --data_dir /path/to/EchoNet-Dynamic\n"
            + kaggle_contents + "\n"
            + "=" * 65
        )

    if videos_dir is None:
        dataset_folder = os.path.dirname(file_list_path)
        adj_videos = os.path.join(dataset_folder, "Videos")
        if os.path.exists(adj_videos):
            videos_dir = adj_videos
        else:
            raise FileNotFoundError(
                f"FileList.csv found at {file_list_path} but no Videos/ "
                f"directory was found alongside it.  Ensure the full "
                f"EchoNet-Dynamic dataset is attached."
            )

    print(f"[Dataset] FileList.csv:       {file_list_path}")
    print(f"[Dataset] VolumeTracings.csv: {vol_tracings_path}")
    print(f"[Dataset] Videos/:            {videos_dir}")
    return file_list_path, vol_tracings_path, videos_dir


# ---------------------------------------------------------------------------
# Plotting utility
# ---------------------------------------------------------------------------

def plot_training_curves(history: list, output_path: str) -> None:
    """Plot and save training / validation loss and MAE curves."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_mae = [h["val_mae"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    ax1.plot(epochs, train_loss, "b-o", label="Train Loss (Smooth L1)")
    ax1.plot(epochs, val_loss, "r-s", label="Val Loss (Smooth L1)")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Loss", fontweight="bold")
    ax1.set_title("Training & Validation Loss", pad=8)
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(epochs, val_mae, "g-^", label="Val MAE (%)")
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("MAE (%)", fontweight="bold")
    ax2.set_title("Validation MAE Progression", pad=8)
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Training curves saved: {output_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    vid_cfg = VideoConfig()
    trn_cfg = TrainingConfig()

    ap = argparse.ArgumentParser(
        description="EchoNet-Dynamic R(2+1)D-18 LVEF Baseline Training on GPU"
    )
    ap.add_argument(
        "--data_dir", type=str, default="",
        help="Path to EchoNet-Dynamic dataset root (contains FileList.csv, Videos/).",
    )
    ap.add_argument(
        "--output_dir", type=str,
        default="/kaggle/working/outputs" if os.path.exists("/kaggle/working") else "./outputs",
    )
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=trn_cfg.batch_size)
    ap.add_argument("--lr", type=float, default=trn_cfg.learning_rate)
    ap.add_argument("--weight_decay", type=float, default=trn_cfg.weight_decay)
    ap.add_argument("--patience", type=int, default=trn_cfg.early_stopping_patience)
    ap.add_argument("--num_workers", type=int, default=trn_cfg.num_workers)
    ap.add_argument("--seed", type=int, default=trn_cfg.seed)
    ap.add_argument(
        "--resume", action="store_true",
        help="Resume training from <output_dir>/last_checkpoint.pt if present.",
    )
    ap.add_argument(
        "--skip_training", action="store_true",
        help="Skip training and run final evaluation on an existing checkpoint.",
    )
    ap.add_argument(
        "--num_test_clips", type=int, default=4,
        help="Number of clips per video for multi-clip test evaluation.",
    )
    args = ap.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )

    print("=" * 65)
    print("ECHONET-DYNAMIC R(2+1)D-18 LVEF BASELINE TRAINING")
    print(f"Device:       {device} ({gpu_name})")
    print(f"Output dir:   {args.output_dir}")
    print(f"Epochs:       {args.epochs}  |  Batch size: {args.batch_size}")
    print(f"Workers:      {args.num_workers} |  Resume: {args.resume}")
    print(f"LR:           {args.lr}      |  Weight decay: {args.weight_decay}")
    print(f"Test clips:   {args.num_test_clips}")
    print("=" * 65)

    # 1. Locate dataset — raises FileNotFoundError with Kaggle instructions if missing
    file_list_csv, vol_tracings_csv, videos_dir = find_dataset_components(args.data_dir)

    # 2. Parse splits (strict patient-level separation via official Split column)
    echo_parser = EchoNetParser(file_list_csv, vol_tracings_csv)
    train_df = echo_parser.get_split("TRAIN")
    val_df = echo_parser.get_split("VAL")
    test_df = echo_parser.get_split("TEST")
    print(
        f"[Dataset] Split sizes — "
        f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}"
    )

    effective_bs = min(args.batch_size, max(1, len(train_df)))

    # 3. Build Datasets and DataLoaders using VideoConfig constants
    shared_kw = dict(
        num_frames=vid_cfg.num_frames,
        frame_stride=vid_cfg.frame_stride,
        image_size=vid_cfg.image_size,
        padding=vid_cfg.padding,
        mean=vid_cfg.mean,
        std=vid_cfg.std,
    )

    train_ds = EchoNetVideoDataset(
        df=train_df, videos_dir=videos_dir, is_train=True, **shared_kw
    )
    val_ds = EchoNetVideoDataset(
        df=val_df, videos_dir=videos_dir, is_train=False, **shared_kw
    )
    test_ds = EchoNetMultiClipDataset(
        df=test_df,
        videos_dir=videos_dir,
        num_clips=args.num_test_clips,
        num_frames=vid_cfg.num_frames,
        frame_stride=vid_cfg.frame_stride,
        image_size=vid_cfg.image_size,
        mean=vid_cfg.mean,
        std=vid_cfg.std,
    )

    loader_kw = dict(
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_ds, batch_size=effective_bs, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=effective_bs, shuffle=False, **loader_kw)
    test_bs = max(1, effective_bs // max(1, args.num_test_clips))
    test_loader = DataLoader(test_ds, batch_size=test_bs, shuffle=False, **loader_kw)

    # 4. Model: R(2+1)D-18 with Kinetics-400 pre-training
    try:
        model = EchoNetR2Plus1D(pretrained=True, dropout_rate=0.2).to(device)
        print("[Model] Loaded with Kinetics-400 pre-trained weights.")
    except Exception as exc:
        print(f"[Model] Pre-trained weights unavailable ({exc}). Using random init.")
        model = EchoNetR2Plus1D(pretrained=False, dropout_rate=0.2).to(device)

    criterion = nn.SmoothL1Loss(beta=trn_cfg.smooth_l1_beta)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=trn_cfg.lr_min
    )
    checkpoint_path = os.path.join(args.output_dir, "best_r2plus1d_lvef.pt")

    # 5. Training
    if not args.skip_training:
        trainer = BaselineTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpoint_dir=args.output_dir,
            patience=args.patience,
            use_amp=True,
        )
        train_result = trainer.train(max_epochs=args.epochs, resume=args.resume)
        if train_result["history"]:
            plot_training_curves(
                train_result["history"],
                os.path.join(args.output_dir, "training_curves.png"),
            )

    # 6. Load best checkpoint for final test evaluation
    print("\n" + "=" * 65)
    print("FINAL EVALUATION — OFFICIAL TEST SET")
    print("=" * 65)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}.\n"
            "Run training first (omit --skip_training) or provide a valid .pt file."
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(
        f"[Eval] Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
        f"(val MAE: {ckpt.get('best_val_mae', float('nan')):.2f}%)"
    )

    # 7. DataLoader-parallelised multi-clip test evaluation
    test_metrics, y_true, y_pred = evaluate_multiclip(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    print("\n[TEST SET RESULTS]")
    print(f"  MAE:                     {test_metrics['mae']:.2f}%")
    print(f"  RMSE:                    {test_metrics['rmse']:.2f}%")
    print(
        f"  Pearson r:               {test_metrics['pearson_r']:.3f} "
        f"(p={test_metrics['pearson_p']:.2e})"
    )
    print(f"  R²:                      {test_metrics['r2']:.3f}")
    print(f"  Bland-Altman bias:       {test_metrics['bland_altman_bias']:.2f}%")
    print(
        f"  Bland-Altman LoA:        "
        f"[{test_metrics['bland_altman_lower_loa']:.2f}%, "
        f"{test_metrics['bland_altman_upper_loa']:.2f}%]"
    )
    print(
        f"  Low-EF (<50%) accuracy:  "
        f"{test_metrics['binary_low_ef_accuracy']*100:.1f}%  "
        f"Sens: {test_metrics['binary_low_ef_sensitivity']*100:.1f}%  "
        f"Spec: {test_metrics['binary_low_ef_specificity']*100:.1f}%"
    )
    print(f"  Test sample count:       {test_metrics['sample_count']}")

    metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(test_metrics, fh, indent=2)
    print(f"\n[Eval] Metrics saved: {metrics_path}")

    plot_scatter_correlation(
        y_true, y_pred,
        save_path=os.path.join(args.output_dir, "test_scatter.png"),
    )
    plot_bland_altman(
        y_true, y_pred,
        save_path=os.path.join(args.output_dir, "test_bland_altman.png"),
    )

    print(f"\nAll artefacts saved to: {args.output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
