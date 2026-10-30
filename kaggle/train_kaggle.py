import os
import sys
import glob
import argparse
import json
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


def find_dataset_root(hint_path: str = None) -> str:
    """Auto-detect EchoNet-Dynamic dataset root on Kaggle or local environments."""
    candidates = []
    if hint_path:
        candidates.append(hint_path)

    # Standard Kaggle input directories
    candidates.extend([
        "/kaggle/input/echonet-dynamic/EchoNet-Dynamic",
        "/kaggle/input/echonet-dynamic",
        "/kaggle/input/stanford-echonet-dynamic/EchoNet-Dynamic",
        "/kaggle/input/stanford-echonet-dynamic",
        "./data/EchoNet-Dynamic",
        "./data"
    ])
    # Search any matching subdirectory under /kaggle/input
    if os.path.exists("/kaggle/input"):
        matches = glob.glob("/kaggle/input/**/FileList.csv", recursive=True)
        for m in matches:
            candidates.insert(0, os.path.dirname(m))

    for path in candidates:
        if path and os.path.exists(os.path.join(path, "FileList.csv")):
            print(f"[Dataset] Found EchoNet-Dynamic root at: {path}")
            return path

    raise FileNotFoundError(
        "Could not automatically locate 'FileList.csv'. "
        "Please provide --data_dir pointing to the folder containing FileList.csv and Videos."
    )


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
    print(f"============================================================")

    # 1. Locate dataset and parse metadata
    data_root = find_dataset_root(args.data_dir)
    file_list_csv = os.path.join(data_root, "FileList.csv")
    vol_tracings_csv = os.path.join(data_root, "VolumeTracings.csv")
    videos_dir = os.path.join(data_root, "Videos")

    parser_meta = EchoNetParser(file_list_csv, vol_tracings_csv)
    train_df = parser_meta.get_split("TRAIN")
    val_df = parser_meta.get_split("VAL")
    test_df = parser_meta.get_split("TEST")

    print(f"[Dataset] Patient Split Counts: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

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
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    # 3. Initialize Model, Loss, Optimizer, Scheduler
    model = EchoNetR2Plus1D(pretrained=True, dropout_rate=0.2).to(device)
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
