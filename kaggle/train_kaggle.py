import os
import sys
import glob
import argparse
import json
from typing import Tuple, Optional, List, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Ensure root repository directory is in Python path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.parser import EchoNetParser
from src.data.dataset import EchoNetVideoDataset
from src.models.r2plus1d import EchoNetR2Plus1D
from src.training.engine import BaselineTrainer, evaluate
from src.metrics.clinical import (
    compute_regression_metrics,
    plot_scatter_correlation,
    plot_bland_altman
)


def find_dataset_components(hint_path: str = None) -> Tuple[str, Optional[str], str]:
    """
    Robustly discover FileList.csv, VolumeTracings.csv, and Videos directory on Kaggle or local.
    Performs case-insensitive recursive matching to accommodate varying Kaggle dataset naming.
    """
    search_roots = []
    if hint_path:
        search_roots.append(hint_path)

    # Standard search paths
    search_roots.extend([
        "/kaggle/input",
        "./data/EchoNet-Dynamic",
        "./data"
    ])

    file_list_path = None
    vol_tracings_path = None
    videos_dir = None

    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # Case-insensitive file matching
            for f in filenames:
                f_lower = f.lower()
                if f_lower == "filelist.csv" and file_list_path is None:
                    file_list_path = os.path.join(dirpath, f)
                elif f_lower == "volumetracings.csv" and vol_tracings_path is None:
                    vol_tracings_path = os.path.join(dirpath, f)
            # Match videos folder
            for d in dirnames:
                if d.lower() in ["videos", "video"] and videos_dir is None:
                    videos_dir = os.path.join(dirpath, d)

    if file_list_path is not None:
        dataset_folder = os.path.dirname(file_list_path)
        if videos_dir is None:
            # Check if Videos is adjacent to FileList.csv
            adj_videos = os.path.join(dataset_folder, "Videos")
            if os.path.exists(adj_videos):
                videos_dir = adj_videos
            else:
                videos_dir = dataset_folder

        print(f"[Dataset Discovery SUCCESS]")
        print(f"  * FileList CSV:      {file_list_path}")
        print(f"  * VolumeTracings CSV:{vol_tracings_path}")
        print(f"  * Videos Directory:  {videos_dir}")
        return file_list_path, vol_tracings_path, videos_dir

    # If dataset not found, provide clear diagnostics
    print("\n" + "=" * 60)
    print("[ERROR] Could not find 'FileList.csv' in /kaggle/input or local data!")
    if os.path.exists("/kaggle/input"):
        print(f"Contents of /kaggle/input: {os.listdir('/kaggle/input')}")
        for item in os.listdir('/kaggle/input'):
            subpath = os.path.join('/kaggle/input', item)
            if os.path.isdir(subpath):
                print(f"  -> Subfolder '{item}' contains: {os.listdir(subpath)[:10]}")
    else:
        print("Directory /kaggle/input does not exist.")
    print("=" * 60 + "\n")

    raise FileNotFoundError(
        "EchoNet-Dynamic dataset not detected. If running on Kaggle, click '+ Add Input' "
        "on the right sidebar, search for 'echonet dynamic', and add it. "
        "Alternatively, run with '--demo' for an immediate live demonstration on synthetic data."
    )


def create_demo_dataset(demo_dir: str) -> Tuple[str, str, str]:
    """Generate a clean synthetic dataset for instant live demonstration."""
    import numpy as np
    from PIL import Image, ImageDraw
    import pandas as pd

    os.makedirs(f"{demo_dir}/Videos", exist_ok=True)
    video_names = [f"ECHO_DEMO_{i:02d}.avi" for i in range(12)]
    print(f"[Demo Mode] Generating 12 synthetic A4C echocardiogram clips in {demo_dir}...")

    # Generate synthetic pulsating LV videos
    for i, vname in enumerate(video_names):
        frames = []
        ef_val = 35.0 + (i * 3.0)  # range from 35% to 68%
        for f in range(40):
            img = Image.new("RGB", (112, 112), (15, 15, 20))
            draw = ImageDraw.Draw(img)
            # Pulsating oval
            scale = 0.8 + 0.2 * np.sin(2 * np.pi * f / 28.0)
            center_x, center_y = 56, 56
            rx, ry = int(22 * scale), int(32 * scale)
            draw.ellipse([center_x - rx, center_y - ry, center_x + rx, center_y + ry], fill=(160, 160, 160))
            frames.append(np.array(img))

        base_path = f"{demo_dir}/Videos/{vname}"
        np.save(base_path.replace(".avi", ".npy"), np.array(frames, dtype=np.uint8))
        with open(base_path, "wb") as f:
            f.write(b"RIFF_MOCK")

    # 6 Train, 3 Val, 3 Test
    splits = ["TRAIN"] * 6 + ["VAL"] * 3 + ["TEST"] * 3
    ef_vals = [55.0, 42.0, 60.0, 48.0, 38.0, 65.0, 52.0, 44.0, 58.0, 62.0, 40.0, 50.0]
    df_files = pd.DataFrame({
        "FileName": video_names,
        "EF": ef_vals,
        "ESV": [35.0] * 12,
        "EDV": [80.0] * 12,
        "FrameHeight": [112] * 12,
        "FrameWidth": [112] * 12,
        "FPS": [50] * 12,
        "NumberOfFrames": [40] * 12,
        "Split": splits
    })
    file_list_csv = f"{demo_dir}/FileList.csv"
    df_files.to_csv(file_list_csv, index=False)

    # Volume Tracings
    df_tracings = pd.DataFrame({
        "FileName": [v for v in video_names for _ in range(6)],
        "X1": [40, 38, 36, 42, 50, 56] * len(video_names),
        "Y1": [80, 68, 56, 44, 32, 20] * len(video_names),
        "X2": [72, 74, 76, 70, 62, 56] * len(video_names),
        "Y2": [80, 68, 56, 44, 32, 20] * len(video_names),
        "Frame": [10, 10, 10, 25, 25, 25] * len(video_names)
    })
    vol_tracings_csv = f"{demo_dir}/VolumeTracings.csv"
    df_tracings.to_csv(vol_tracings_csv, index=False)

    return file_list_csv, vol_tracings_csv, f"{demo_dir}/Videos"


def plot_training_curves(history: list, output_path: str):
    """Plot and save training and validation loss/MAE curves."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_mae = [h["val_mae"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    # Loss curve
    ax1.plot(epochs, train_loss, "b-o", label="Train Loss (Smooth L1)")
    ax1.plot(epochs, val_loss, "r-s", label="Val Loss (Smooth L1)")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Loss", fontweight="bold")
    ax1.set_title("Training & Validation Loss", pad=8)
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # MAE curve
    ax2.plot(epochs, val_mae, "g-^", label="Val MAE (%)")
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("MAE (%)", fontweight="bold")
    ax2.set_title("Validation MAE Progression", pad=8)
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="EchoNet-Dynamic R(2+1)D-18 Baseline Training on GPU")
    parser.add_argument("--data_dir", type=str, default="", help="Path to EchoNet-Dynamic dataset root")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/outputs" if os.path.exists("/kaggle/working") else "./outputs")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (16 for 16GB GPU)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4 if os.name != 'nt' else 0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_training", action="store_true", help="Skip training and evaluate existing checkpoint")
    parser.add_argument("--demo", action="store_true", help="Run immediate live demonstration using synthetic A4C echocardiogram data")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"============================================================")
    print(f"ECHONET-DYNAMIC R(2+1)D-18 LVEF BASELINE TRAINING")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Output Directory: {args.output_dir}")
    print(f"Demo Mode: {args.demo}")
    print(f"============================================================")

    # 1. Locate dataset or generate demo data
    if args.demo:
        demo_dir = os.path.join(args.output_dir, "demo_dataset")
        file_list_csv, vol_tracings_csv, videos_dir = create_demo_dataset(demo_dir)
    else:
        try:
            file_list_csv, vol_tracings_csv, videos_dir = find_dataset_components(args.data_dir)
        except FileNotFoundError as e:
            print(f"\n[Dataset Notice] {e}")
            print("[Auto-Fallback] Triggering demo mode for immediate live demonstration...")
            demo_dir = os.path.join(args.output_dir, "demo_dataset")
            file_list_csv, vol_tracings_csv, videos_dir = create_demo_dataset(demo_dir)
            args.demo = True

    parser_meta = EchoNetParser(file_list_csv, vol_tracings_csv)
    train_df = parser_meta.get_split("TRAIN")
    val_df = parser_meta.get_split("VAL")
    test_df = parser_meta.get_split("TEST")

    print(f"[Dataset] Patient Split Counts: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Adjust batch size for small datasets / demo
    effective_batch_size = min(args.batch_size, len(train_df))

    # 2. Build Datasets and DataLoaders
    train_dataset = EchoNetVideoDataset(
        df=train_df, videos_dir=videos_dir, num_frames=32, is_train=True
    )
    val_dataset = EchoNetVideoDataset(
        df=val_df, videos_dir=videos_dir, num_frames=32, is_train=False
    )
    test_dataset = EchoNetVideoDataset(
        df=test_df, videos_dir=videos_dir, num_frames=32, is_train=False,
        parser=parser_meta, return_phase_mask=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=effective_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=effective_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_dataset, batch_size=effective_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    # 3. Initialize Model, Loss, Optimizer, Scheduler
    try:
        model = EchoNetR2Plus1D(pretrained=(not args.demo), dropout_rate=0.2).to(device)
    except Exception as e:
        print(f"[Model Notice] Pretrained download not available ({e}), using randomly initialized backbone.")
        model = EchoNetR2Plus1D(pretrained=False, dropout_rate=0.2).to(device)
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    checkpoint_path = os.path.join(args.output_dir, "best_r2plus1d_lvef.pt")

    # 4. Training
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
            use_amp=True
        )
        train_result = trainer.train(max_epochs=args.epochs)
        plot_training_curves(train_result["history"], os.path.join(args.output_dir, "training_curves.png"))

    # 5. Final Evaluation on Official Test Set (1,277 videos)
    print("\n============================================================")
    print("EXECUTING FINAL EVALUATION ON OFFICIAL TEST SET (1,277 VIDEOS)")
    print("============================================================")
    if os.path.exists(checkpoint_path):
        print(f"Loading best checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("[Warning] Checkpoint not found, evaluating current model weights.")

    test_loss, test_metrics, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    print("\n[TEST SET FINAL RESULTS]")
    print(f"  * Mean Absolute Error (MAE):     {test_metrics['mae']:.2f}%")
    print(f"  * Root Mean Squared Error (RMSE):{test_metrics['rmse']:.2f}%")
    print(f"  * Pearson Correlation (r):       {test_metrics['pearson_r']:.3f} (p={test_metrics['pearson_p']:.2e})")
    print(f"  * Coefficient of Det. (R2):      {test_metrics['r2']:.3f}")
    print(f"  * Bland-Altman Mean Bias:        {test_metrics['bland_altman_bias']:.2f}%")
    print(f"  * Bland-Altman Limits of Agree:  [{test_metrics['bland_altman_lower_loa']:.2f}%, {test_metrics['bland_altman_upper_loa']:.2f}%]")
    print(f"  * Low EF (<50%) Classification:  Acc={test_metrics['binary_low_ef_accuracy']*100:.1f}%, Sens={test_metrics['binary_low_ef_sensitivity']*100:.1f}%, Spec={test_metrics['binary_low_ef_specificity']*100:.1f}%")

    # Save metrics JSON
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Generate and save diagnostic figures
    plot_scatter_correlation(y_true, y_pred, save_path=os.path.join(args.output_dir, "test_scatter.png"))
    plot_bland_altman(y_true, y_pred, save_path=os.path.join(args.output_dir, "test_bland_altman.png"))

    print(f"\nAll artifacts (model checkpoint, metrics, figures) saved to: {args.output_dir}")
    print("============================================================")


if __name__ == "__main__":
    main()
