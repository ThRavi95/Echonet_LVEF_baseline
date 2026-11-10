import os
import sys
import shutil
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import EchoNetMultiClipDataset, EchoNetVideoDataset
from src.data.parser import EchoNetParser
from src.models.r2plus1d import EchoNetR2Plus1D
from src.training.engine import BaselineTrainer, evaluate, evaluate_multiclip
from kaggle.train_kaggle import plot_training_curves


def create_synthetic_mock_data(test_dir: str):
    """Generate minimal synthetic dataset for rapid training smoke testing."""
    os.makedirs(f"{test_dir}/Videos", exist_ok=True)
    video_names = [
        "TRAIN_01.avi", "TRAIN_02.avi",
        "VAL_01.avi", "VAL_02.avi",
        "TEST_01.avi", "TEST_02.avi"
    ]

    for vname in video_names:
        frames = []
        for f in range(35):
            img = Image.new("RGB", (112, 112), (20, 20, 25))
            draw = ImageDraw.Draw(img)
            draw.ellipse([35, 30, 75, 80], fill=(160, 160, 160))
            frames.append(np.array(img))

        base_path = f"{test_dir}/Videos/{vname}"
        np.save(base_path.replace(".avi", ".npy"), np.array(frames, dtype=np.uint8))
        with open(base_path, "wb") as f:
            f.write(b"RIFF_MOCK")

    df_files = pd.DataFrame({
        "FileName": video_names,
        "EF": [60.0, 45.0, 52.0, 48.0, 58.0, 39.0],
        "ESV": [30.0, 50.0, 40.0, 45.0, 35.0, 55.0],
        "EDV": [75.0, 90.0, 85.0, 88.0, 80.0, 92.0],
        "FrameHeight": [112] * 6,
        "FrameWidth": [112] * 6,
        "FPS": [50] * 6,
        "NumberOfFrames": [35] * 6,
        "Split": ["TRAIN", "TRAIN", "VAL", "VAL", "TEST", "TEST"]
    })
    df_files.to_csv(f"{test_dir}/FileList.csv", index=False)


def run_training_smoketest():
    print("=" * 60)
    print("RUNNING TASKS 2.3 & 2.4: TRAINING ENGINE SMOKE TEST")
    print("=" * 60)

    test_dir = "./scratch/train_smoketest_data"
    output_dir = "./scratch/train_smoketest_output"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. Create mock data
        print("[1/5] Creating mock dataset for training smoke test...")
        create_synthetic_mock_data(test_dir)

        # 2. Parse data and construct loaders
        print("[2/5] Initializing datasets and DataLoaders...")
        parser = EchoNetParser(f"{test_dir}/FileList.csv")
        train_df = parser.get_split("TRAIN")
        val_df = parser.get_split("VAL")
        test_df = parser.get_split("TEST")

        train_ds = EchoNetVideoDataset(train_df, f"{test_dir}/Videos", num_frames=32, is_train=True)
        val_ds = EchoNetVideoDataset(val_df, f"{test_dir}/Videos", num_frames=32, is_train=False)
        test_ds = EchoNetMultiClipDataset(test_df, f"{test_dir}/Videos", num_clips=2, num_frames=32)

        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        # 3. Initialize Model and Trainer
        print("[3/5] Instantiating Model, Loss, Optimizer, and BaselineTrainer...")
        device = torch.device("cpu")
        model = EchoNetR2Plus1D(pretrained=False, dropout_rate=0.2).to(device)
        criterion = nn.SmoothL1Loss(beta=1.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)

        trainer = BaselineTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpoint_dir=output_dir,
            patience=3,
            use_amp=False
        )

        # 4. Train for 1 epoch + verify last_checkpoint.pt + resume to epoch 2
        print("[4/5] Executing 1 training epoch + resume smoke test...")
        results = trainer.train(max_epochs=1)

        best_ckpt = results["best_checkpoint"]
        last_ckpt = results["last_checkpoint"]
        assert os.path.exists(best_ckpt), f"Best checkpoint not found at: {best_ckpt}"
        assert os.path.exists(last_ckpt), f"Last checkpoint not found at: {last_ckpt}"
        print(f" -> Best & Last checkpoints verified at: {output_dir}")

        resumed = trainer.train(max_epochs=2, resume=True)
        assert len(resumed["history"]) == 2, f"Expected 2 history entries after resume, got {len(resumed['history'])}"
        print(" -> Resume from last_checkpoint.pt verified (epoch 1 -> epoch 2)!")

        plot_training_curves(resumed["history"], f"{output_dir}/curves.png")
        assert os.path.exists(f"{output_dir}/curves.png"), "Training curves figure not created"
        print(" -> Training curves plotted and verified!")

        # 5. Multi-clip evaluate on test set
        print("[5/5] Evaluating checkpoint on test set with evaluate_multiclip...")
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics, y_true, y_pred = evaluate_multiclip(model, test_loader, device)

        print(f" -> Multi-clip Test MAE: {test_metrics['mae']:.2f}%")
        assert test_metrics["mae"] >= 0.0, "MAE must be non-negative"
        assert len(y_pred) == len(test_df), "Prediction count mismatch"

        print("=" * 60)
        print("SUCCESS: TRAINING ENGINE AND KAGGLE PIPELINE VERIFIED!")
        print("=" * 60)

    finally:
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


if __name__ == "__main__":
    run_training_smoketest()
