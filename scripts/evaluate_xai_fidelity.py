"""
Quantitative XAI Explanation Fidelity Evaluation Pipeline
===========================================================
Measures the spatial and temporal explanation fidelity of the 3D Grad-CAM
saliency maps produced by the EchoNet R(2+1)D-18 continuous regression model
against expert cardiologist Left Ventricular (LV) endocardial volume tracings
at End-Diastole (ED) and End-Systole (ES).

Publication-grade implementation adhering to:
  - Ouyang et al. (Nature, 2020) - Stanford EchoNet-Dynamic benchmark
  - Hu et al. (J. Cardiovasc. Imaging, 2024) - XAI in Echocardiography
  - Zhang et al. (Frontiers in Med., 2024) - Quantitative anatomical alignment

Metrics Evaluated:
  - Intersection over Union (IoU / Jaccard Index) at ED and ES
  - Sørensen-Dice Coefficient (F1 Score) at ED and ES
  - Pointing Game Hit / Accuracy (Peak saliency localization inside LV chamber)
  - Attention Energy Fraction (Continuous saliency mass inside chamber)
  - Correlation between Explanation Fidelity (IoU) and LVEF Prediction Error

Usage:
  python scripts/evaluate_xai_fidelity.py \\
      --checkpoint outputs/best_r2plus1d_lvef.pt \\
      --data_dir ./data/EchoNet-Dynamic \\
      --output_dir outputs \\
      --threshold 0.5 \\
      --save_visualizations
"""
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import VideoConfig
from src.data.io import get_video_total_frames, load_video_clip
from src.data.parser import EchoNetParser
from src.data.rasterizer import rasterize_lv_mask
from src.metrics.explainability import GradCAM3D
from src.metrics.iou import compute_comprehensive_fidelity_metrics
from src.models.r2plus1d import EchoNetR2Plus1D


# ---------------------------------------------------------------------------
# Dataset & Model Discovery Helpers
# ---------------------------------------------------------------------------

def find_dataset_components(hint_path: str = "") -> Tuple[str, Optional[str], str]:
    """
    Locate FileList.csv, VolumeTracings.csv, and Videos/ folder.
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
        raise FileNotFoundError(
            "Could not locate FileList.csv. Please specify --data_dir pointing "
            "to the EchoNet-Dynamic dataset."
        )

    if videos_dir is None:
        dataset_folder = os.path.dirname(file_list_path)
        adj_videos = os.path.join(dataset_folder, "Videos")
        if os.path.exists(adj_videos):
            videos_dir = adj_videos
        else:
            raise FileNotFoundError(
                f"Videos/ folder not found alongside {file_list_path}."
            )

    return file_list_path, vol_tracings_path, videos_dir


def load_model_checkpoint(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load EchoNetR2Plus1D architecture and populate trained weights from checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}\n"
            "Ensure the model has finished training or provide a valid path."
        )

    model = EchoNetR2Plus1D(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Resolve state_dict structure
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strip potential 'module.' prefixes from DataParallel training
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()
    return model


def resolve_video_file_path(videos_dir: str, filename: str) -> Optional[str]:
    """
    Resolve video file path handling case mismatches and missing extensions.
    """
    direct_path = os.path.join(videos_dir, filename)
    if os.path.exists(direct_path):
        return direct_path

    # Check companion .npy for ARM64/local environments
    stem = os.path.splitext(filename)[0]
    npy_path = os.path.join(videos_dir, f"{stem}.npy")
    if os.path.exists(npy_path):
        return npy_path

    # Try lowercase and uppercase .avi
    lower_avi = os.path.join(videos_dir, f"{stem}.avi")
    if os.path.exists(lower_avi):
        return lower_avi

    upper_avi = os.path.join(videos_dir, f"{stem}.AVI")
    if os.path.exists(upper_avi):
        return upper_avi

    # Case-insensitive filesystem scan fallback
    stem_lower = stem.lower()
    try:
        for entry in os.scandir(videos_dir):
            entry_stem = os.path.splitext(entry.name)[0].lower()
            if entry_stem == stem_lower:
                return entry.path
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Temporal Clip Extraction & Phase Alignment
# ---------------------------------------------------------------------------

def extract_phase_aligned_clip(
    video_path: str,
    ed_frame: int,
    es_frame: int,
    num_frames: int = 32,
    stride: int = 2,
    target_size: int = 112,
) -> Tuple[torch.Tensor, int, float, int]:
    """
    Extract a 32-frame video clip temporally aligned to encompass the cardiac
    phases (ED and ES).

    Returns:
        video_tensor: (1, 3, 32, 112, 112) normalized tensor
        t_ed_idx: integer index of ED frame within the 32-frame sampled clip
        t_es_idx: float index of ES frame within the 32-frame sampled clip
        start_frame: raw frame index in video where clip begins
    """
    total_frames = get_video_total_frames(video_path)
    span = num_frames * stride  # 64 frames

    # Center the sampling window around the midpoint of ED and ES
    mid_frame = (ed_frame + es_frame) // 2
    ideal_start = mid_frame - span // 2

    # Constrain within video duration
    if total_frames <= span:
        start_frame = 0
    else:
        start_frame = max(0, min(ideal_start, total_frames - span))

    # Fine-tune start_frame parity to match ed_frame modulo stride if possible
    ed_offset = ed_frame - start_frame
    if ed_offset % stride != 0 and start_frame > 0:
        start_frame -= 1
    elif ed_offset % stride != 0 and (start_frame + span) < total_frames:
        start_frame += 1

    # Load normalized clip: shape (32, 112, 112, 3)
    clip_np = load_video_clip(
        video_path=video_path,
        start_frame=start_frame,
        num_frames=num_frames,
        stride=stride,
        target_size=target_size,
    )

    # Convert to PyTorch (1, 3, 32, 112, 112)
    video_tensor = torch.from_numpy(clip_np).permute(3, 0, 1, 2).unsqueeze(0).float()

    # Compute temporal indices within the sampled clip
    t_ed_idx = int(np.clip((ed_frame - start_frame) // stride, 0, num_frames - 1))
    t_es_idx = float(np.clip((es_frame - start_frame) / float(stride), 0.0, num_frames - 1.0))

    return video_tensor, t_ed_idx, t_es_idx, start_frame


def interpolate_temporal_slice(heatmap_3d: torch.Tensor, t_float: float) -> np.ndarray:
    """
    Extract a 2D spatial heatmap slice at a continuous temporal index t_float.
    If t_float is fractional (e.g. 12.5), performs linear temporal interpolation
    between adjacent frames.
    """
    T, H, W = heatmap_3d.shape
    t_low = int(np.floor(t_float))
    t_high = int(np.ceil(t_float))

    t_low = max(0, min(T - 1, t_low))
    t_high = max(0, min(T - 1, t_high))

    if t_low == t_high:
        return heatmap_3d[t_low].detach().cpu().numpy()

    alpha = float(t_float - t_low)
    slice_low = heatmap_3d[t_low].detach().cpu().numpy()
    slice_high = heatmap_3d[t_high].detach().cpu().numpy()

    return (1.0 - alpha) * slice_low + alpha * slice_high


# ---------------------------------------------------------------------------
# Visualization Generator
# ---------------------------------------------------------------------------

def save_xai_fidelity_figure(
    output_path: str,
    patient_id: str,
    pred_ef: float,
    true_ef: float,
    ed_slice: np.ndarray,
    ed_mask: np.ndarray,
    ed_heatmap: np.ndarray,
    ed_metrics: Dict[str, float],
    es_slice: np.ndarray,
    es_mask: np.ndarray,
    es_heatmap: np.ndarray,
    es_metrics: Dict[str, float],
) -> None:
    """
    Save a multi-panel publication-grade figure comparing Grad-CAM spatial
    attributions to ground-truth LV endocardial tracings at ED and ES.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Top-Left: ED Ground Truth Mask Overlay
    axes[0, 0].imshow(ed_slice, cmap="gray")
    axes[0, 0].contour(ed_mask, colors="lime", linewidths=1.5)
    axes[0, 0].set_title(
        f"End-Diastole (ED) — Ground Truth Tracing\nMask Area: {int(ed_mask.sum())} px",
        fontsize=11,
        fontweight="bold",
    )
    axes[0, 0].axis("off")

    # Top-Right: ED Grad-CAM Saliency Overlay
    axes[0, 1].imshow(ed_slice, cmap="gray")
    axes[0, 1].imshow(ed_heatmap, cmap="jet", alpha=0.5)
    axes[0, 1].contour(ed_mask, colors="white", linewidths=1.2, linestyles="dashed")
    axes[0, 1].set_title(
        f"ED 3D Grad-CAM Attribution\nIoU: {ed_metrics['iou']:.3f} | Dice: {ed_metrics['dice']:.3f} | Hit: {int(ed_metrics['pointing_game_hit'])}",
        fontsize=11,
        fontweight="bold",
    )
    axes[0, 1].axis("off")

    # Bottom-Left: ES Ground Truth Mask Overlay
    axes[1, 0].imshow(es_slice, cmap="gray")
    axes[1, 0].contour(es_mask, colors="red", linewidths=1.5)
    axes[1, 0].set_title(
        f"End-Systole (ES) — Ground Truth Tracing\nMask Area: {int(es_mask.sum())} px",
        fontsize=11,
        fontweight="bold",
    )
    axes[1, 0].axis("off")

    # Bottom-Right: ES Grad-CAM Saliency Overlay
    axes[1, 1].imshow(es_slice, cmap="gray")
    axes[1, 1].imshow(es_heatmap, cmap="jet", alpha=0.5)
    axes[1, 1].contour(es_mask, colors="white", linewidths=1.2, linestyles="dashed")
    axes[1, 1].set_title(
        f"ES 3D Grad-CAM Attribution\nIoU: {es_metrics['iou']:.3f} | Dice: {es_metrics['dice']:.3f} | Hit: {int(es_metrics['pointing_game_hit'])}",
        fontsize=11,
        fontweight="bold",
    )
    axes[1, 1].axis("off")

    # Super Title
    abs_err = abs(pred_ef - true_ef)
    fig.suptitle(
        f"Patient: {patient_id} | True LVEF: {true_ef:.1f}% | Pred LVEF: {pred_ef:.1f}% (Error: {abs_err:.1f}%)\n"
        f"Mean IoU Fidelity: {(ed_metrics['iou'] + es_metrics['iou'])/2.0:.3f}",
        fontsize=13,
        fontweight="heavy",
        y=0.98,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main Evaluation Execution
# ---------------------------------------------------------------------------

def run_xai_fidelity_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Executes the quantitative explanation fidelity pipeline across the TEST split.
    """
    print("\n" + "=" * 70)
    print("  EchoNet-Dynamic Quantitative XAI Explanation Fidelity Pipeline")
    print("  Objective 4: Grad-CAM 3D Anatomical Alignment at ED and ES")
    print("=" * 70)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"[*] Execution device: {device}")

    # 1. Discover dataset
    file_list_csv, vol_tracings_csv, videos_dir = find_dataset_components(args.data_dir)
    print(f"[*] FileList.csv:        {file_list_csv}")
    print(f"[*] VolumeTracings.csv:  {vol_tracings_csv}")
    print(f"[*] Videos Directory:    {videos_dir}")

    if not vol_tracings_csv or not os.path.exists(vol_tracings_csv):
        raise FileNotFoundError(
            "VolumeTracings.csv is required to compute ground-truth anatomical masks. "
            "Please ensure VolumeTracings.csv is located in the dataset directory."
        )

    # 2. Parse dataset
    parser = EchoNetParser(file_list_csv=file_list_csv, volume_tracings_csv=vol_tracings_csv)
    test_df = parser.get_split("TEST")
    print(f"[*] Total TEST split videos: {len(test_df)}")

    # 3. Load trained model
    print(f"[*] Loading model checkpoint from: {args.checkpoint}")
    model = load_model_checkpoint(args.checkpoint, device)
    print(f"[*] Successfully loaded EchoNetR2Plus1D model.")

    # 4. Initialize 3D Grad-CAM
    grad_cam = GradCAM3D(model=model, target_layer=model.layer4)
    print(f"[*] GradCAM3D initialized hooking into 'model.layer4'.")

    # 5. Iteration over test split
    per_patient_records: List[Dict[str, Any]] = []
    skipped_count = 0
    vis_saved = 0

    max_samples = args.max_samples if args.max_samples and args.max_samples > 0 else len(test_df)
    eval_df = test_df.iloc[:max_samples]

    pbar = tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating XAI Fidelity")

    for _, row in pbar:
        filename = str(row["FileName"])
        true_ef = float(row["EF"])

        # Resolve video path
        video_path = resolve_video_file_path(videos_dir, filename)
        if not video_path:
            skipped_count += 1
            continue

        # Lookup cardiac phase frames
        ed_frame, es_frame = parser.get_cardiac_phase_frames(filename)
        if ed_frame is None or es_frame is None:
            skipped_count += 1
            continue

        # Rasterize ground-truth LV masks at 112x112
        ed_coords = parser.get_tracing_coordinates(filename, ed_frame)
        es_coords = parser.get_tracing_coordinates(filename, es_frame)

        if not ed_coords or not es_coords:
            skipped_count += 1
            continue

        ed_mask = rasterize_lv_mask(
            ed_coords["x1"], ed_coords["y1"], ed_coords["x2"], ed_coords["y2"],
            orig_shape=(112, 112), target_shape=(112, 112)
        )
        es_mask = rasterize_lv_mask(
            es_coords["x1"], es_coords["y1"], es_coords["x2"], es_coords["y2"],
            orig_shape=(112, 112), target_shape=(112, 112)
        )

        # Skip degenerate empty masks
        if ed_mask.sum() == 0 or es_mask.sum() == 0:
            skipped_count += 1
            continue

        # Extract aligned clip
        try:
            video_tensor, t_ed, t_es, start_frame = extract_phase_aligned_clip(
                video_path=video_path,
                ed_frame=ed_frame,
                es_frame=es_frame,
                num_frames=32,
                stride=2,
                target_size=112,
            )
        except Exception:
            skipped_count += 1
            continue

        # Run 3D Grad-CAM
        try:
            heatmap_3d, pred_ef = grad_cam.generate_heatmap_with_prediction(video_tensor)
        except Exception:
            skipped_count += 1
            continue

        # Slice 2D heatmaps at ED and ES
        heatmap_ed = heatmap_3d[t_ed].detach().cpu().numpy()
        heatmap_es = interpolate_temporal_slice(heatmap_3d, t_es)

        # Compute comprehensive fidelity metrics
        metrics_ed = compute_comprehensive_fidelity_metrics(
            heatmap_ed, ed_mask, threshold=args.threshold
        )
        metrics_es = compute_comprehensive_fidelity_metrics(
            heatmap_es, es_mask, threshold=args.threshold
        )

        mean_iou = (metrics_ed["iou"] + metrics_es["iou"]) / 2.0
        mean_dice = (metrics_ed["dice"] + metrics_es["dice"]) / 2.0
        mean_hit = (metrics_ed["pointing_game_hit"] + metrics_es["pointing_game_hit"]) / 2.0
        mean_energy = (metrics_ed["energy_fraction"] + metrics_es["energy_fraction"]) / 2.0

        abs_error = abs(pred_ef - true_ef)

        record = {
            "FileName": filename,
            "True_EF": round(true_ef, 2),
            "Pred_EF": round(pred_ef, 2),
            "EF_Abs_Error": round(abs_error, 2),
            "ED_Frame": int(ed_frame),
            "ES_Frame": int(es_frame),
            "ED_IoU": round(metrics_ed["iou"], 4),
            "ES_IoU": round(metrics_es["iou"], 4),
            "Mean_IoU": round(mean_iou, 4),
            "ED_Dice": round(metrics_ed["dice"], 4),
            "ES_Dice": round(metrics_es["dice"], 4),
            "Mean_Dice": round(mean_dice, 4),
            "ED_Pointing_Hit": int(metrics_ed["pointing_game_hit"]),
            "ES_Pointing_Hit": int(metrics_es["pointing_game_hit"]),
            "Mean_Pointing_Hit": round(mean_hit, 4),
            "ED_Energy_Fraction": round(metrics_ed["energy_fraction"], 4),
            "ES_Energy_Fraction": round(metrics_es["energy_fraction"], 4),
            "Mean_Energy_Fraction": round(mean_energy, 4),
        }
        per_patient_records.append(record)

        # Save visualization if requested
        if args.save_visualizations and vis_saved < args.vis_count:
            # Reconstruct un-normalized video frames for visualization
            raw_clip = video_tensor.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
            raw_clip = np.clip(raw_clip * 0.18 + 0.12, 0.0, 1.0)
            ed_frame_img = raw_clip[t_ed]
            t_es_int = int(round(t_es))
            es_frame_img = raw_clip[min(t_es_int, 31)]

            vis_path = os.path.join(
                args.output_dir, "xai_visualizations", f"{os.path.splitext(filename)[0]}_fidelity.png"
            )
            save_xai_fidelity_figure(
                output_path=vis_path,
                patient_id=filename,
                pred_ef=pred_ef,
                true_ef=true_ef,
                ed_slice=ed_frame_img,
                ed_mask=ed_mask,
                ed_heatmap=heatmap_ed,
                ed_metrics=metrics_ed,
                es_slice=es_frame_img,
                es_mask=es_mask,
                es_heatmap=heatmap_es,
                es_metrics=metrics_es,
            )
            vis_saved += 1

        pbar.set_postfix({"Mean_IoU": f"{mean_iou:.3f}", "MAE": f"{abs_error:.1f}%"})

    grad_cam.remove_hooks()

    if len(per_patient_records) == 0:
        raise RuntimeError("No test videos could be evaluated. Check dataset paths and annotations.")

    # 6. Aggregate population statistics
    ed_ious = np.array([r["ED_IoU"] for r in per_patient_records])
    es_ious = np.array([r["ES_IoU"] for r in per_patient_records])
    mean_ious = np.array([r["Mean_IoU"] for r in per_patient_records])

    ed_dices = np.array([r["ED_Dice"] for r in per_patient_records])
    es_dices = np.array([r["ES_Dice"] for r in per_patient_records])
    mean_dices = np.array([r["Mean_Dice"] for r in per_patient_records])

    ed_hits = np.array([r["ED_Pointing_Hit"] for r in per_patient_records])
    es_hits = np.array([r["ES_Pointing_Hit"] for r in per_patient_records])
    mean_hits = np.array([r["Mean_Pointing_Hit"] for r in per_patient_records])

    ed_energies = np.array([r["ED_Energy_Fraction"] for r in per_patient_records])
    es_energies = np.array([r["ES_Energy_Fraction"] for r in per_patient_records])
    mean_energies = np.array([r["Mean_Energy_Fraction"] for r in per_patient_records])

    ef_errors = np.array([r["EF_Abs_Error"] for r in per_patient_records])

    # Pearson & Spearman correlation between IoU and Absolute LVEF Error
    pearson_r = 0.0
    spearman_r = 0.0
    if len(mean_ious) > 1 and np.std(mean_ious) > 1e-6 and np.std(ef_errors) > 1e-6:
        pearson_r = float(np.corrcoef(mean_ious, ef_errors)[0, 1])
        try:
            from scipy.stats import spearmanr
            spearman_r = float(spearmanr(mean_ious, ef_errors).statistic)
        except Exception:
            spearman_r = pearson_r

    summary_metrics = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_samples": len(eval_df),
        "evaluated_samples": len(per_patient_records),
        "skipped_samples": skipped_count,
        "binarization_threshold": args.threshold,
        "mean_absolute_error_lvef": float(np.mean(ef_errors)),
        "rmse_lvef": float(np.sqrt(np.mean(ef_errors ** 2))),
        "ed_metrics": {
            "iou_mean": float(np.mean(ed_ious)),
            "iou_median": float(np.median(ed_ious)),
            "iou_std": float(np.std(ed_ious)),
            "dice_mean": float(np.mean(ed_dices)),
            "dice_median": float(np.median(ed_dices)),
            "pointing_game_accuracy": float(np.mean(ed_hits)),
            "energy_fraction_mean": float(np.mean(ed_energies)),
        },
        "es_metrics": {
            "iou_mean": float(np.mean(es_ious)),
            "iou_median": float(np.median(es_ious)),
            "iou_std": float(np.std(es_ious)),
            "dice_mean": float(np.mean(es_dices)),
            "dice_median": float(np.median(es_dices)),
            "pointing_game_accuracy": float(np.mean(es_hits)),
            "energy_fraction_mean": float(np.mean(es_energies)),
        },
        "overall_fidelity": {
            "iou_mean": float(np.mean(mean_ious)),
            "iou_median": float(np.median(mean_ious)),
            "iou_std": float(np.std(mean_ious)),
            "dice_mean": float(np.mean(mean_dices)),
            "dice_median": float(np.median(mean_dices)),
            "pointing_game_accuracy": float(np.mean(mean_hits)),
            "energy_fraction_mean": float(np.mean(mean_energies)),
            "pearson_correlation_iou_vs_ef_error": float(pearson_r),
            "spearman_correlation_iou_vs_ef_error": float(spearman_r),
        },
        "per_patient_results": per_patient_records,
    }

    # 7. Export outputs
    os.makedirs(args.output_dir, exist_ok=True)
    out_json_path = os.path.join(args.output_dir, "xai_metrics.json")
    with open(out_json_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)

    # 8. Print formatted report
    print("\n" + "=" * 70)
    print("  QUANTITATIVE XAI FIDELITY BENCHMARK REPORT (Ouyang et al. / EchoNet)")
    print("=" * 70)
    print(f"  Total Test Evaluated:       {len(per_patient_records)}")
    print(f"  Binarization Cutoff:        {args.threshold}")
    print(f"  LVEF Mean Absolute Error:   {summary_metrics['mean_absolute_error_lvef']:.2f}%")
    print(f"  LVEF Root Mean Sq Error:    {summary_metrics['rmse_lvef']:.2f}%")
    print("-" * 70)
    print(f"  End-Diastole (ED) Mean IoU: {summary_metrics['ed_metrics']['iou_mean']:.4f} (Median: {summary_metrics['ed_metrics']['iou_median']:.4f})")
    print(f"  End-Diastole (ED) Dice:     {summary_metrics['ed_metrics']['dice_mean']:.4f}")
    print(f"  End-Diastole Pointing Hit:  {summary_metrics['ed_metrics']['pointing_game_accuracy']*100:.1f}%")
    print(f"  End-Diastole Energy Frac:   {summary_metrics['ed_metrics']['energy_fraction_mean']*100:.1f}%")
    print("-" * 70)
    print(f"  End-Systole (ES) Mean IoU:  {summary_metrics['es_metrics']['iou_mean']:.4f} (Median: {summary_metrics['es_metrics']['iou_median']:.4f})")
    print(f"  End-Systole (ES) Dice:      {summary_metrics['es_metrics']['dice_mean']:.4f}")
    print(f"  End-Systole Pointing Hit:   {summary_metrics['es_metrics']['pointing_game_accuracy']*100:.1f}%")
    print(f"  End-Systole Energy Frac:    {summary_metrics['es_metrics']['energy_fraction_mean']*100:.1f}%")
    print("-" * 70)
    print(f"  OVERALL MEAN IoU:           {summary_metrics['overall_fidelity']['iou_mean']:.4f}")
    print(f"  OVERALL MEAN DICE:          {summary_metrics['overall_fidelity']['dice_mean']:.4f}")
    print(f"  OVERALL POINTING HIT:       {summary_metrics['overall_fidelity']['pointing_game_accuracy']*100:.1f}%")
    print(f"  OVERALL ATTENTION ENERGY:   {summary_metrics['overall_fidelity']['energy_fraction_mean']*100:.1f}%")
    print(f"  Pearson Correlation (IoU vs Error):  r = {summary_metrics['overall_fidelity']['pearson_correlation_iou_vs_ef_error']:.3f}")
    print(f"  Spearman Correlation (IoU vs Error): rho = {summary_metrics['overall_fidelity']['spearman_correlation_iou_vs_ef_error']:.3f}")
    print("=" * 70)
    print(f"[✓] Full quantitative XAI metrics exported to: {out_json_path}")
    if args.save_visualizations:
        print(f"[✓] Top-{vis_saved} patient visual overlays saved to: {os.path.join(args.output_dir, 'xai_visualizations')}")

    return summary_metrics


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantitative XAI Explanation Fidelity Evaluation for EchoNet R(2+1)D-18"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/best_r2plus1d_lvef.pt",
        help="Path to trained PyTorch model weights (.pt checkpoint).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Path to EchoNet-Dynamic dataset root (contains FileList.csv, VolumeTracings.csv, Videos/).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save xai_metrics.json and visual overlays.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarization cutoff threshold applied to normalized Grad-CAM heatmaps (default: 0.5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Execution device ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum test samples to evaluate (0 = evaluate entire TEST split).",
    )
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save high-resolution 4-panel visual overlays for individual test cases.",
    )
    parser.add_argument(
        "--vis_count",
        type=int,
        default=5,
        help="Number of patient visualization figures to save.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_xai_fidelity_evaluation(cli_args)
