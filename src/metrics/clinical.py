import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from scipy import stats


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard clinical and statistical regression metrics for LVEF.
    
    Args:
        y_true: Ground truth LVEF % array of shape (N,).
        y_pred: Predicted LVEF % array of shape (N,).
        
    Returns:
        Dict containing MAE, RMSE, Pearson r, R2, Mean Bias, and Limits of Agreement.
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})")
    if len(y_true) < 2:
        raise ValueError("At least 2 samples required to compute clinical metrics.")

    # 1. Residuals
    diffs = y_pred - y_true
    
    # 2. MAE & RMSE
    mae = float(np.mean(np.abs(diffs)))
    rmse = float(np.sqrt(np.mean(diffs ** 2)))

    # 3. Pearson Correlation (r) and p-value
    r, p_val = stats.pearsonr(y_true, y_pred)
    
    # 4. Coefficient of Determination (R2)
    ss_res = np.sum(diffs ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # 5. Bland-Altman Limits of Agreement
    mean_bias = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))
    upper_loa = float(mean_bias + 1.96 * sd_diff)
    lower_loa = float(mean_bias - 1.96 * sd_diff)

    # 6. Binary Classification of Low EF (< 50%)
    true_low = (y_true < 50.0)
    pred_low = (y_pred < 50.0)
    acc_binary = float(np.mean(true_low == pred_low))
    
    tp = np.sum(true_low & pred_low)
    fp = np.sum((~true_low) & pred_low)
    fn = np.sum(true_low & (~pred_low))
    tn = np.sum((~true_low) & (~pred_low))
    
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "pearson_r": float(r),
        "pearson_p": float(p_val),
        "r2": r2,
        "bland_altman_bias": mean_bias,
        "bland_altman_upper_loa": upper_loa,
        "bland_altman_lower_loa": lower_loa,
        "binary_low_ef_accuracy": acc_binary,
        "binary_low_ef_sensitivity": sensitivity,
        "binary_low_ef_specificity": specificity,
        "sample_count": len(y_true)
    }


def plot_scatter_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Generate publication-ready scatter plot with line of identity and regression line."""
    metrics = compute_regression_metrics(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="none", c="#1f77b4", s=25)
    
    # 45-degree line of identity
    min_val = min(y_true.min(), y_pred.min()) - 5
    max_val = max(y_true.max(), y_pred.max()) + 5
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.75, label="Identity (y=x)")
    
    # Best fit regression line
    m, b = np.polyfit(y_true, y_pred, 1)
    ax.plot(y_true, m * y_true + b, color="#d62728", lw=1.5, label=f"Fit (slope={m:.2f})")

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("True LVEF (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted LVEF (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"LVEF Correlation: r={metrics['pearson_r']:.3f}, MAE={metrics['mae']:.2f}%",
        fontsize=12, pad=10
    )
    ax.legend(frameon=True, loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Generate publication-ready Bland-Altman agreement plot."""
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    
    means = (y_true + y_pred) / 2.0
    diffs = y_pred - y_true
    
    bias = np.mean(diffs)
    sd = np.std(diffs, ddof=1)
    upper_loa = bias + 1.96 * sd
    lower_loa = bias - 1.96 * sd

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.scatter(means, diffs, alpha=0.5, c="#2ca02c", s=25, edgecolors="none")

    # Reference horizontal lines
    ax.axhline(bias, color="blue", linestyle="-", lw=1.5, label=f"Bias: {bias:.2f}%")
    ax.axhline(upper_loa, color="red", linestyle="--", lw=1.2, label=f"+1.96 SD: {upper_loa:.2f}%")
    ax.axhline(lower_loa, color="red", linestyle="--", lw=1.2, label=f"-1.96 SD: {lower_loa:.2f}%")

    ax.set_xlabel("Average LVEF: (True + Predicted) / 2 (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Difference: (Predicted - True) (%)", fontsize=11, fontweight="bold")
    ax.set_title("Bland-Altman Limits of Agreement for LVEF", fontsize=12, pad=10)
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
