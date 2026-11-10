"""
Ultrasound Video Degradation & Robustness Stress Benchmarking Pipeline
======================================================================
Jointly benchmarks:
  1. Continuous LVEF Prediction Robustness (MAE, RMSE)
  2. Spatiotemporal XAI Explanation Fidelity (IoU, Dice, Pointing Hit)

across controlled acoustic degradation stress conditions:
  • Clean (Uncorrupted baseline)
  • Multiplicative Speckle Noise (eta ~ N(0, 0.25^2))
  • Transducer Sector Acoustic Shadowing (75% attenuation cone)
  • Wireless Frame Dropout (25% temporal drop rate)
  • Compound Clinical POCUS Degradation (Speckle + Shadow + Dropout)

Directly fulfills Master Project Objective 1 and Objective 4.

Usage:
  python scripts/benchmark_degradation_robustness.py \\
      --checkpoint outputs/best_r2plus1d_lvef.pt \\
      --data_dir ./data/EchoNet-Dynamic \\
      --output_dir outputs \\
      --max_samples 50
"""
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.evaluate_xai_fidelity import (
    extract_phase_aligned_clip,
    find_dataset_components,
    interpolate_temporal_slice,
    load_model_checkpoint,
    resolve_video_file_path,
)
from src.data.degradation import UltrasoundDegradation
from src.data.parser import EchoNetParser
from src.data.rasterizer import rasterize_lv_mask
from src.metrics.explainability import GradCAM3D
from src.metrics.iou import compute_comprehensive_fidelity_metrics


def run_robustness_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print("  EchoNet-Dynamic Robustness & XAI Degradation Stress Benchmark")
    print("  Jointly evaluating LVEF Error and Saliency Fidelity under Acoustic Noise")
    print("=" * 75)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"[*] Execution device: {device}")

    # 1. Discover dataset
    file_list_csv, vol_tracings_csv, videos_dir = find_dataset_components(args.data_dir)
    parser = EchoNetParser(file_list_csv=file_list_csv, volume_tracings_csv=vol_tracings_csv)
    test_df = parser.get_split("TEST")
    print(f"[*] Total TEST split videos: {len(test_df)}")

    # 2. Load model & Grad-CAM
    model = load_model_checkpoint(args.checkpoint, device)
    grad_cam = GradCAM3D(model=model, target_layer=model.layer4)
    degrader = UltrasoundDegradation(seed=42)

    max_samples = args.max_samples if args.max_samples and args.max_samples > 0 else len(test_df)
    eval_df = test_df.iloc[:max_samples]
    print(f"[*] Benchmarking on {len(eval_df)} test samples across 5 stress conditions...")

    # Define test conditions
    conditions = [
        "Clean (Baseline)",
        "Speckle Noise (0.25)",
        "Acoustic Shadow (0.75)",
        "Frame Drop (0.25)",
        "Compound POCUS Stress",
    ]

    benchmark_results: Dict[str, Dict[str, Any]] = {
        cond: {
            "ef_errors": [],
            "mean_ious": [],
            "ed_ious": [],
            "es_ious": [],
            "mean_dices": [],
            "pointing_hits": [],
            "energy_fractions": [],
        }
        for cond in conditions
    }

    # Pre-filter valid samples with both video and tracings
    valid_samples: List[Dict[str, Any]] = []
    for _, row in eval_df.iterrows():
        fname = str(row["FileName"])
        vpath = resolve_video_file_path(videos_dir, fname)
        if not vpath:
            continue
        ed_f, es_f = parser.get_cardiac_phase_frames(fname)
        if ed_f is None or es_f is None:
            continue
        ed_coords = parser.get_tracing_coordinates(fname, ed_f)
        es_coords = parser.get_tracing_coordinates(fname, es_f)
        if not ed_coords or not es_coords:
            continue
        ed_mask = rasterize_lv_mask(
            ed_coords["x1"], ed_coords["y1"], ed_coords["x2"], ed_coords["y2"],
            orig_shape=(112, 112), target_shape=(112, 112)
        )
        es_mask = rasterize_lv_mask(
            es_coords["x1"], es_coords["y1"], es_coords["x2"], es_coords["y2"],
            orig_shape=(112, 112), target_shape=(112, 112)
        )
        if ed_mask.sum() == 0 or es_mask.sum() == 0:
            continue
        valid_samples.append({
            "filename": fname,
            "video_path": vpath,
            "true_ef": float(row["EF"]),
            "ed_frame": ed_f,
            "es_frame": es_f,
            "ed_mask": ed_mask,
            "es_mask": es_mask,
        })

    print(f"[*] Valid annotated samples for benchmarking: {len(valid_samples)}")
    if len(valid_samples) == 0:
        raise RuntimeError("No valid annotated samples found.")

    for sample in tqdm(valid_samples, desc="Stress Testing Videos"):
        # Extract clean normalized clip: (1, 3, 32, 112, 112)
        clean_tensor, t_ed, t_es, _ = extract_phase_aligned_clip(
            video_path=sample["video_path"],
            ed_frame=sample["ed_frame"],
            es_frame=sample["es_frame"],
            num_frames=32,
            stride=2,
            target_size=112,
        )

        # Un-normalize to [0, 1] for degradation application: (32, 112, 112, 3)
        raw_clip = clean_tensor.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        raw_clip = np.clip(raw_clip * 0.18 + 0.12, 0.0, 1.0)

        # Generate the 5 variants
        clip_variants = {
            "Clean (Baseline)": raw_clip,
            "Speckle Noise (0.25)": degrader.add_speckle_noise(raw_clip, severity=0.25),
            "Acoustic Shadow (0.75)": degrader.apply_acoustic_shadowing(raw_clip, attenuation_factor=0.75),
            "Frame Drop (0.25)": degrader.apply_frame_dropout(raw_clip, drop_rate=0.25, mode="freeze"),
            "Compound POCUS Stress": degrader.apply_compound_degradation(raw_clip, severity="moderate"),
        }

        for cond_name, variant_np in clip_variants.items():
            # Re-normalize to EchoNet domain: (x - 0.12) / 0.18
            norm_variant = (variant_np.astype(np.float32) - 0.12) / 0.18
            tensor_var = torch.from_numpy(norm_variant).permute(3, 0, 1, 2).unsqueeze(0).float()

            # Run 3D Grad-CAM
            try:
                heatmap_3d, pred_ef = grad_cam.generate_heatmap_with_prediction(tensor_var)
            except Exception:
                continue

            # Slice ED and ES
            hm_ed = heatmap_3d[t_ed].detach().cpu().numpy()
            hm_es = interpolate_temporal_slice(heatmap_3d, t_es)

            # Compute fidelity
            m_ed = compute_comprehensive_fidelity_metrics(hm_ed, sample["ed_mask"], threshold=args.threshold)
            m_es = compute_comprehensive_fidelity_metrics(hm_es, sample["es_mask"], threshold=args.threshold)

            mean_iou = (m_ed["iou"] + m_es["iou"]) / 2.0
            mean_dice = (m_ed["dice"] + m_es["dice"]) / 2.0
            mean_hit = (m_ed["pointing_game_hit"] + m_es["pointing_game_hit"]) / 2.0
            mean_energy = (m_ed["energy_fraction"] + m_es["energy_fraction"]) / 2.0
            abs_err = abs(pred_ef - sample["true_ef"])

            bucket = benchmark_results[cond_name]
            bucket["ef_errors"].append(abs_err)
            bucket["mean_ious"].append(mean_iou)
            bucket["ed_ious"].append(m_ed["iou"])
            bucket["es_ious"].append(m_es["iou"])
            bucket["mean_dices"].append(mean_dice)
            bucket["pointing_hits"].append(mean_hit)
            bucket["energy_fractions"].append(mean_energy)

    grad_cam.remove_hooks()

    # Aggregate condition statistics
    summary_report: Dict[str, Any] = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluated_samples": len(valid_samples),
        "conditions": {},
    }

    print("\n" + "=" * 90)
    print(f"{'Condition':<25} | {'LVEF MAE':<10} | {'RMSE':<8} | {'Mean IoU':<10} | {'Mean Dice':<10} | {'Hit Rate':<10}")
    print("-" * 90)

    for cond_name, metrics in benchmark_results.items():
        errs = np.array(metrics["ef_errors"])
        ious = np.array(metrics["mean_ious"])
        dices = np.array(metrics["mean_dices"])
        hits = np.array(metrics["pointing_hits"])
        energies = np.array(metrics["energy_fractions"])

        mae = float(np.mean(errs)) if len(errs) > 0 else 0.0
        rmse = float(np.sqrt(np.mean(errs ** 2))) if len(errs) > 0 else 0.0
        mean_iou = float(np.mean(ious)) if len(ious) > 0 else 0.0
        mean_dice = float(np.mean(dices)) if len(dices) > 0 else 0.0
        hit_rate = float(np.mean(hits)) if len(hits) > 0 else 0.0
        energy_frac = float(np.mean(energies)) if len(energies) > 0 else 0.0

        summary_report["conditions"][cond_name] = {
            "lvef_mae": round(mae, 2),
            "lvef_rmse": round(rmse, 2),
            "mean_iou": round(mean_iou, 4),
            "mean_dice": round(mean_dice, 4),
            "pointing_game_accuracy": round(hit_rate, 4),
            "attention_energy_fraction": round(energy_frac, 4),
        }

        print(f"{cond_name:<25} | {mae:<9.2f}% | {rmse:<7.2f}% | {mean_iou:<10.4f} | {mean_dice:<10.4f} | {hit_rate*100:<9.1f}%")

    print("=" * 90)

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "robustness_benchmark.json")
    with open(out_file, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"[✓] Benchmark report exported to: {out_file}")

    return summary_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EchoNet Robustness & XAI Degradation Benchmark")
    parser.add_argument("--checkpoint", type=str, default="outputs/best_r2plus1d_lvef.pt")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    run_robustness_benchmark(parse_args())

