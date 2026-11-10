"""
Spatial Overlap & Explanation Fidelity Metrics
===============================================
Computes quantitative alignment metrics between continuous 2D/3D saliency heatmaps
(e.g., Grad-CAM 3D temporal slices) and anatomical ground-truth segmentation masks
(e.g., Left Ventricular endocardial contours at End-Diastole and End-Systole).

Metrics implemented:
  • Intersection over Union (IoU / Jaccard Index) with configurable binarization threshold
  • Sørensen–Dice Coefficient (Dice / F1 Score)
  • Pointing Game Hit / Accuracy (argmax peak localization inside anatomical boundary)
  • Attention Energy Fraction (continuous probability mass located inside chamber)
"""
from typing import Dict, Tuple, Union

import numpy as np
import torch


def _to_numpy_2d(arr: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Ensure array is a 2D float32 numpy array."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        arr = arr.squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array or tensor, got shape {arr.shape}.")
    return arr


def compute_heatmap_iou(
    heatmap_2d: Union[torch.Tensor, np.ndarray],
    true_mask_2d: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
) -> float:
    """
    Compute Intersection over Union (IoU / Jaccard index) between a continuous
    heatmap and a binary ground-truth segmentation mask.

    Args:
        heatmap_2d: (H, W) array or tensor with normalized values in [0.0, 1.0].
        true_mask_2d: (H, W) binary or boolean mask (1 = inside anatomical contour, 0 = outside).
        threshold: Binarization cutoff applied to the continuous heatmap.

    Returns:
        float: Scalar IoU in range [0.0, 1.0].
               If both binarized prediction and ground-truth are empty (union == 0),
               returns 1.0 (perfect null agreement).
               If union == 0 but one is non-empty, returns 0.0.
    """
    h_arr = _to_numpy_2d(heatmap_2d)
    m_arr = _to_numpy_2d(true_mask_2d)

    if h_arr.shape != m_arr.shape:
        raise ValueError(
            f"Dimension mismatch between heatmap {h_arr.shape} and mask {m_arr.shape}."
        )

    pred_bin = h_arr >= threshold
    true_bin = m_arr > 0.5

    intersection = float(np.logical_and(pred_bin, true_bin).sum())
    union = float(np.logical_or(pred_bin, true_bin).sum())

    if union == 0.0:
        # Both binarized prediction and ground-truth are completely empty
        return 1.0 if (pred_bin.sum() == 0 and true_bin.sum() == 0) else 0.0

    return float(intersection / union)


def compute_dice_coefficient(
    heatmap_2d: Union[torch.Tensor, np.ndarray],
    true_mask_2d: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
) -> float:
    """
    Compute the Sørensen–Dice coefficient (F1 score) between a binarized heatmap
    and a ground-truth segmentation mask: 2 * |A ∩ B| / (|A| + |B|).

    Args:
        heatmap_2d: (H, W) array or tensor with normalized values in [0.0, 1.0].
        true_mask_2d: (H, W) binary or boolean mask.
        threshold: Binarization cutoff applied to heatmap.

    Returns:
        float: Dice coefficient in [0.0, 1.0].
    """
    h_arr = _to_numpy_2d(heatmap_2d)
    m_arr = _to_numpy_2d(true_mask_2d)

    if h_arr.shape != m_arr.shape:
        raise ValueError(
            f"Dimension mismatch between heatmap {h_arr.shape} and mask {m_arr.shape}."
        )

    pred_bin = h_arr >= threshold
    true_bin = m_arr > 0.5

    intersection = float(np.logical_and(pred_bin, true_bin).sum())
    sum_cardinalities = float(pred_bin.sum() + true_bin.sum())

    if sum_cardinalities == 0.0:
        return 1.0

    return float(2.0 * intersection / sum_cardinalities)


def compute_pointing_game_hit(
    heatmap_2d: Union[torch.Tensor, np.ndarray],
    true_mask_2d: Union[torch.Tensor, np.ndarray],
) -> bool:
    """
    Evaluate the Pointing Game hit condition: checks whether the global maximum
    pixel (peak saliency) of the heatmap falls strictly inside the ground-truth mask.

    Args:
        heatmap_2d: (H, W) continuous heatmap.
        true_mask_2d: (H, W) binary segmentation mask.

    Returns:
        bool: True if argmax(heatmap) lies inside true_mask_2d > 0.5, False otherwise.
    """
    h_arr = _to_numpy_2d(heatmap_2d)
    m_arr = _to_numpy_2d(true_mask_2d)

    if h_arr.shape != m_arr.shape:
        raise ValueError(
            f"Dimension mismatch between heatmap {h_arr.shape} and mask {m_arr.shape}."
        )

    # In case of multiple equal max values, take the first occurrence
    max_idx = np.unravel_index(np.argmax(h_arr, axis=None), h_arr.shape)
    return bool(m_arr[max_idx] > 0.5)


def compute_attention_energy_fraction(
    heatmap_2d: Union[torch.Tensor, np.ndarray],
    true_mask_2d: Union[torch.Tensor, np.ndarray],
    epsilon: float = 1e-8,
) -> float:
    """
    Compute the fraction of total continuous heatmap energy located inside
    the anatomical chamber mask: sum_{(i,j) in mask} H(i,j) / (sum_{all} H(i,j) + eps).

    This provides a threshold-free measure of saliency localization fidelity.

    Args:
        heatmap_2d: (H, W) continuous heatmap.
        true_mask_2d: (H, W) binary segmentation mask.
        epsilon: Small positive float to prevent division by zero.

    Returns:
        float: Scalar fraction in [0.0, 1.0].
    """
    h_arr = np.clip(_to_numpy_2d(heatmap_2d), 0.0, None)
    m_arr = (_to_numpy_2d(true_mask_2d) > 0.5).astype(np.float32)

    total_energy = float(h_arr.sum())
    if total_energy < epsilon:
        return 0.0

    chamber_energy = float((h_arr * m_arr).sum())
    return float(np.clip(chamber_energy / (total_energy + epsilon), 0.0, 1.0))


def compute_comprehensive_fidelity_metrics(
    heatmap_2d: Union[torch.Tensor, np.ndarray],
    true_mask_2d: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all standard quantitative XAI fidelity metrics in a single call.

    Returns:
        Dict containing: 'iou', 'dice', 'pointing_game_hit', 'energy_fraction'.
    """
    iou = compute_heatmap_iou(heatmap_2d, true_mask_2d, threshold=threshold)
    dice = compute_dice_coefficient(heatmap_2d, true_mask_2d, threshold=threshold)
    pointing_hit = 1.0 if compute_pointing_game_hit(heatmap_2d, true_mask_2d) else 0.0
    energy_frac = compute_attention_energy_fraction(heatmap_2d, true_mask_2d)

    return {
        "iou": iou,
        "dice": dice,
        "pointing_game_hit": pointing_hit,
        "energy_fraction": energy_frac,
    }

