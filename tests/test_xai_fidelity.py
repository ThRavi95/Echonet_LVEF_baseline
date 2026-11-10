"""
Unit Tests for Quantitative XAI Explainability & Fidelity Metrics
=================================================================
Tests:
  1. GradCAM3D hook registration, gradient propagation, and spatiotemporal upsampling
  2. IoU, Dice, Pointing Game, and Attention Energy Fraction mathematical correctness
  3. Edge cases: all-zero heatmaps, completely mismatched masks, identical masks
  4. Phase-aligned temporal slice interpolation in evaluate_xai_fidelity
"""
import numpy as np
import torch
import torch.nn as nn

from src.metrics.explainability import GradCAM3D
from src.metrics.iou import (
    compute_attention_energy_fraction,
    compute_comprehensive_fidelity_metrics,
    compute_dice_coefficient,
    compute_heatmap_iou,
    compute_pointing_game_hit,
)
from src.models.r2plus1d import EchoNetR2Plus1D
from scripts.evaluate_xai_fidelity import interpolate_temporal_slice


def test_iou_identical_masks():
    """Identical binary masks with threshold=0.5 must yield IoU = 1.0 and Dice = 1.0."""
    mask = np.zeros((112, 112), dtype=np.float32)
    mask[30:70, 30:70] = 1.0  # 40x40 square

    heatmap = np.zeros((112, 112), dtype=np.float32)
    heatmap[30:70, 30:70] = 0.8  # Above 0.5 threshold

    iou = compute_heatmap_iou(heatmap, mask, threshold=0.5)
    dice = compute_dice_coefficient(heatmap, mask, threshold=0.5)

    assert np.isclose(iou, 1.0, atol=1e-5)
    assert np.isclose(dice, 1.0, atol=1e-5)


def test_iou_disjoint_masks():
    """Completely disjoint regions must yield IoU = 0.0 and Dice = 0.0."""
    mask = np.zeros((112, 112), dtype=np.float32)
    mask[10:30, 10:30] = 1.0

    heatmap = np.zeros((112, 112), dtype=np.float32)
    heatmap[60:80, 60:80] = 1.0

    iou = compute_heatmap_iou(heatmap, mask, threshold=0.5)
    dice = compute_dice_coefficient(heatmap, mask, threshold=0.5)

    assert iou == 0.0
    assert dice == 0.0


def test_iou_partial_overlap():
    """
    Two 10x10 squares overlapping on a 10x5 region:
    Intersection = 50, Union = 100 + 100 - 50 = 150 -> IoU = 50/150 = 0.3333...
    Dice = 2 * 50 / (100 + 100) = 0.5
    """
    mask = np.zeros((112, 112), dtype=np.float32)
    mask[10:20, 10:20] = 1.0

    heatmap = np.zeros((112, 112), dtype=np.float32)
    heatmap[10:20, 15:25] = 0.9

    iou = compute_heatmap_iou(heatmap, mask, threshold=0.5)
    dice = compute_dice_coefficient(heatmap, mask, threshold=0.5)

    assert np.isclose(iou, 50.0 / 150.0, atol=1e-4)
    assert np.isclose(dice, 0.5, atol=1e-4)


def test_pointing_game_hit_and_miss():
    """Pointing game should return True when peak is in mask, False otherwise."""
    mask = np.zeros((112, 112), dtype=np.float32)
    mask[40:60, 40:60] = 1.0

    # Case 1: Hit (peak inside mask)
    heatmap_hit = np.full((112, 112), 0.1, dtype=np.float32)
    heatmap_hit[50, 50] = 1.0  # inside
    assert compute_pointing_game_hit(heatmap_hit, mask) is True

    # Case 2: Miss (peak outside mask)
    heatmap_miss = np.full((112, 112), 0.1, dtype=np.float32)
    heatmap_miss[10, 10] = 1.0  # outside
    assert compute_pointing_game_hit(heatmap_miss, mask) is False


def test_attention_energy_fraction():
    """Energy fraction should match ratio of within-chamber energy to total energy."""
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[0:5, 0:5] = 1.0  # 25 pixels

    heatmap = np.zeros((10, 10), dtype=np.float32)
    heatmap[0:5, 0:5] = 2.0  # within mask energy = 25 * 2 = 50
    heatmap[5:10, 5:10] = 2.0  # outside mask energy = 25 * 2 = 50
    # Total = 100, Fraction = 50 / 100 = 0.5

    frac = compute_attention_energy_fraction(heatmap, mask)
    assert np.isclose(frac, 0.5, atol=1e-4)


def test_gradcam3d_shape_and_range():
    """GradCAM3D must produce (32, 112, 112) heatmap with values normalized to [0, 1]."""
    model = EchoNetR2Plus1D(pretrained=False)
    grad_cam = GradCAM3D(model=model, target_layer=model.layer4)

    dummy_video = torch.randn(1, 3, 32, 112, 112)
    heatmap, pred_ef = grad_cam.generate_heatmap_with_prediction(dummy_video)

    assert heatmap.shape == (32, 112, 112)
    assert isinstance(pred_ef, float)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0

    grad_cam.remove_hooks()


def test_interpolate_temporal_slice():
    """Check continuous temporal interpolation between heatmap slices."""
    heatmap = torch.zeros(32, 112, 112)
    heatmap[10] = 2.0
    heatmap[11] = 4.0

    # Midpoint at t=10.5 should be exactly (2.0 + 4.0)/2 = 3.0
    interp = interpolate_temporal_slice(heatmap, 10.5)
    assert np.allclose(interp, 3.0)

    # Exact integer at t=10.0 should be 2.0
    interp_exact = interpolate_temporal_slice(heatmap, 10.0)
    assert np.allclose(interp_exact, 2.0)


def test_ultrasound_degradation():
    """Verify speckle noise, acoustic shadow, and frame drop transformations."""
    from src.data.degradation import UltrasoundDegradation

    degrader = UltrasoundDegradation(seed=42)
    video = np.full((32, 112, 112, 3), 0.5, dtype=np.float32)

    # 1. Speckle noise should introduce variance around 0.5
    noisy = degrader.add_speckle_noise(video, severity=0.2)
    assert noisy.shape == video.shape
    assert not np.allclose(noisy, video)
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0

    # 2. Acoustic shadowing should attenuate signal in a central sector
    shadowed = degrader.apply_acoustic_shadowing(video, attenuation_factor=0.8, center_angle_deg=0.0)
    assert shadowed.shape == video.shape
    assert shadowed.min() < 0.5  # Signal is attenuated inside shadow cone
    assert shadowed.max() <= 0.5  # Outside shadow remains at most original 0.5

    # 3. Frame dropout should copy/freeze frames
    dropped = degrader.apply_frame_dropout(video, drop_rate=0.5, mode="freeze")
    assert dropped.shape == video.shape

    # 4. PyTorch tensor support
    tensor_video = torch.full((1, 3, 32, 112, 112), 0.5, dtype=torch.float32)
    compound = degrader.apply_compound_degradation(tensor_video, severity="moderate")
    assert isinstance(compound, torch.Tensor)
    assert compound.shape == tensor_video.shape


def test_clinical_report_generator():
    """Verify clinical classification, fidelity trust scoring, and NLP narrative."""
    from src.clinical.report import ClinicalReportGenerator

    gen = ClinicalReportGenerator()

    # Case 1: Normal LVEF (60%) with High Fidelity (IoU = 0.5)
    rep_normal = gen.generate_report(
        patient_id="0X001",
        predicted_ef=60.0,
        true_ef=58.0,
        iou_ed=0.52,
        iou_es=0.48,
        pointing_hit=True,
    )
    assert rep_normal["classification"]["severity"] == "Normal"
    assert rep_normal["fidelity"]["fidelity_level"] == "High Anatomical Fidelity"
    assert rep_normal["fidelity"]["spurious_attention_warning"] is False
    assert "Normal LV Systolic Function" in rep_normal["markdown_report"]

    # Case 2: Severe LVEF (24%) with Low Fidelity (IoU = 0.15) -> ICD alert + spurious warning
    rep_severe = gen.generate_report(
        patient_id="0X002",
        predicted_ef=24.0,
        true_ef=22.0,
        iou_ed=0.15,
        iou_es=0.10,
        pointing_hit=False,
    )
    assert rep_severe["classification"]["severity"] == "Severe"
    assert rep_severe["fidelity"]["fidelity_level"] == "Low Anatomical Fidelity"
    assert rep_severe["fidelity"]["spurious_attention_warning"] is True
    assert len(rep_severe["clinical_alerts"]) >= 2  # ICD warning + Low fidelity warning


def test_lvef_estimator_pipeline(tmp_path=None):
    """Verify LVEFEstimator initialization, checkpoint loading, and explain method."""
    import tempfile
    import os
    from src.inference import LVEFEstimator

    # Save a temporary dummy model checkpoint
    model = EchoNetR2Plus1D(pretrained=False)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_ckpt:
        ckpt_path = tmp_ckpt.name
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    # Save a companion dummy .npy video: (64, 112, 112, 3) uint8
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_video:
        video_path = tmp_video.name
        dummy_frames = np.full((64, 112, 112, 3), 128, dtype=np.uint8)
        np.save(video_path, dummy_frames)

    try:
        estimator = LVEFEstimator(checkpoint_path=ckpt_path, device="cpu", num_clips=2)

        # 1. Test multi-clip prediction
        result = estimator.predict(video_path)
        assert "predicted_lvef" in result
        assert "clinical_category" in result
        assert "ase_classification" in result
        assert "summary" in result
        assert 10.0 <= result["predicted_lvef"] <= 85.0

        # 2. Test 3D Grad-CAM explanation
        raw_frames, heatmap_3d, clip_ef = estimator.explain(video_path, start_frame=0)
        assert raw_frames.shape == (32, 112, 112, 3)
        assert heatmap_3d.shape == (32, 112, 112)
        assert isinstance(clip_ef, float)
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        if os.path.exists(video_path):
            os.remove(video_path)


