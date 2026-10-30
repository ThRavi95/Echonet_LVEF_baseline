import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.r2plus1d import EchoNetR2Plus1D
from src.metrics.clinical import (
    compute_regression_metrics,
    plot_scatter_correlation,
    plot_bland_altman
)


def run_tests():
    print("=" * 60)
    print("RUNNING TASKS 2.1 & 2.2: MODEL ARCHITECTURE & METRICS VERIFICATION")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. Test Model Instantiation & Forward Pass (Task 2.1)
    # -------------------------------------------------------------
    print("[1/4] Instantiating EchoNetR2Plus1D (pretrained=False for CPU test)...")
    model = EchoNetR2Plus1D(pretrained=False, dropout_rate=0.2)
    model.eval()

    # Create dummy video batch: (Batch=2, Channels=3, Frames=32, Height=112, Width=112)
    dummy_input = torch.randn(2, 3, 32, 112, 112, dtype=torch.float32)
    print(f" -> Input tensor shape: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f" -> Forward pass output: {output}")
    print(f" -> Output shape: {output.shape} (Expected: [2])")
    assert output.shape == torch.Size([2]), f"Expected shape [2], got {output.shape}"
    assert not torch.isnan(output).any(), "Model forward pass produced NaN!"
    print(" -> Model forward pass successfully verified!")

    # -------------------------------------------------------------
    # 2. Test Loss & Backpropagation
    # -------------------------------------------------------------
    print("[2/4] Testing backward pass with Smooth L1 Loss...")
    model.train()
    criterion = nn.SmoothL1Loss(beta=1.0)
    target_ef = torch.tensor([55.0, 42.5], dtype=torch.float32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    preds = model(dummy_input)
    loss = criterion(preds, target_ef)
    loss.backward()
    optimizer.step()

    print(f" -> Backward pass loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    print(" -> Backward pass and optimizer step successfully verified!")

    # Verify Grad-CAM anchor hook
    last_conv = model.get_last_conv_layer()
    assert last_conv is not None, "get_last_conv_layer() returned None"
    print(f" -> Last conv layer hooked: {last_conv.__class__.__name__}")

    # -------------------------------------------------------------
    # 3. Test Clinical Metrics (Task 2.2)
    # -------------------------------------------------------------
    print("[3/4] Testing clinical metrics computation...")
    # Synthetic true and predicted EF vectors
    np.random.seed(42)
    y_true = np.array([62.0, 58.0, 45.0, 35.0, 50.0, 28.0, 65.0, 40.0])
    y_pred = y_true + np.array([2.5, -3.0, 1.2, -2.1, 4.0, 3.5, -1.5, -4.0])

    metrics = compute_regression_metrics(y_true, y_pred)
    for k, v in metrics.items():
        print(f"    - {k}: {v:.4f}" if isinstance(v, float) else f"    - {k}: {v}")

    assert metrics["mae"] > 0, "MAE must be positive"
    assert metrics["rmse"] >= metrics["mae"], "RMSE must be >= MAE"
    assert -1.0 <= metrics["pearson_r"] <= 1.0, "Pearson r must be in [-1, 1]"
    assert metrics["sample_count"] == len(y_true), "Sample count mismatch"
    print(" -> Clinical metrics verified!")

    # -------------------------------------------------------------
    # 4. Test Plot Generation
    # -------------------------------------------------------------
    print("[4/4] Testing Bland-Altman and Scatter plot generation...")
    os.makedirs("./scratch", exist_ok=True)
    scatter_fig = plot_scatter_correlation(y_true, y_pred, save_path="./scratch/test_scatter.png")
    ba_fig = plot_bland_altman(y_true, y_pred, save_path="./scratch/test_bland_altman.png")

    assert os.path.exists("./scratch/test_scatter.png"), "Scatter plot not saved"
    assert os.path.exists("./scratch/test_bland_altman.png"), "Bland-Altman plot not saved"
    print(" -> Scatter and Bland-Altman plots generated successfully!")

    print("=" * 60)
    print("SUCCESS: TASKS 2.1 & 2.2 FULLY VERIFIED!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
