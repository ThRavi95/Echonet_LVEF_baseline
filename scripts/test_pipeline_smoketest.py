import os
import sys
import shutil
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

# Add repository root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.rasterizer import rasterize_lv_mask
from src.data.parser import EchoNetParser
from src.data.dataset import EchoNetVideoDataset
from src.data.io import load_video_clip, get_video_total_frames


def create_synthetic_echo_avi(output_path: str, num_frames: int = 45, size: int = 112):
    """Generate a synthetic pulsating apical-4-chamber mockup video and save as .npy and .avi."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []
    for f in range(num_frames):
        # Create dark background
        img = Image.new("RGB", (size, size), (15, 15, 20))
        draw = ImageDraw.Draw(img)

        # Pulsating oval simulating Left Ventricle cavity contraction
        scale = 0.8 + 0.2 * np.sin(2 * np.pi * f / 30.0)
        center_x, center_y = size // 2, size // 2
        r_x, r_y = int(25 * scale), int(35 * scale)
        bbox = [center_x - r_x, center_y - r_y, center_x + r_x, center_y + r_y]
        draw.ellipse(bbox, fill=(180, 180, 180), outline=(255, 255, 255))

        frames.append(np.array(img))

    # Save as .npy companion (works on any OS/platform without ffmpeg binaries)
    npy_path = output_path.replace(".avi", ".npy")
    np.save(npy_path, np.array(frames, dtype=np.uint8))
    # Also create empty/stub .avi so os.path.exists checks pass
    with open(output_path, "wb") as f:
        f.write(b"RIFF_MOCK_AVI")


def run_pipeline_smoketest():
    print("=" * 60)
    print("RUNNING TASK 1.4: DATA PIPELINE CPU SMOKE TEST")
    print("=" * 60)

    test_dir = "./scratch/smoketest_data"
    os.makedirs(f"{test_dir}/Videos", exist_ok=True)

    # 1. Test Rasterizer with synthetic Simpson's points
    print("[1/5] Testing LV Mask Rasterizer...")
    # Simpson's points from base (y=80) to apex (y=20)
    x1 = [40, 38, 36, 42, 50, 56]
    y1 = [80, 68, 56, 44, 32, 20]
    x2 = [72, 74, 76, 70, 62, 56]
    y2 = [80, 68, 56, 44, 32, 20]

    mask = rasterize_lv_mask(x1, y1, x2, y2, orig_shape=(112, 112), target_shape=(112, 112))
    assert mask.shape == (112, 112), f"Expected (112, 112), got {mask.shape}"
    assert mask.sum() > 0, "Rasterized mask should have non-zero pixels inside LV contour"
    print(f" -> LV Mask rasterized successfully! Area: {mask.sum()} pixels.")

    # 2. Create Mock FileList.csv and VolumeTracings.csv
    print("[2/5] Creating Mock EchoNet-Dynamic CSVs & synthetic AVIs...")
    video_filenames = ["MOCK_001.avi", "MOCK_002.avi"]
    for vname in video_filenames:
        create_synthetic_echo_avi(f"{test_dir}/Videos/{vname}")

    df_files = pd.DataFrame({
        "FileName": video_filenames,
        "EF": [58.5, 42.0],
        "ESV": [35.2, 52.0],
        "EDV": [85.0, 90.0],
        "FrameHeight": [112, 112],
        "FrameWidth": [112, 112],
        "FPS": [50, 50],
        "NumberOfFrames": [45, 45],
        "Split": ["TRAIN", "VAL"]
    })
    df_files.to_csv(f"{test_dir}/FileList.csv", index=False)

    # Frame 10 = larger contour (ED, End-Diastole); Frame 25 = smaller contour (ES)
    # Use distinct x1/y1/x2/y2 so area-based ED/ES assignment is unambiguous.
    x1_ed = [30, 28, 30, 36, 46, 56]   # wider polygon → larger area → ED
    y1_ed = [80, 65, 50, 36, 22, 10]
    x2_ed = [82, 84, 82, 76, 66, 56]
    y2_ed = [80, 65, 50, 36, 22, 10]

    x1_es = [42, 41, 42, 46, 52, 56]   # narrower polygon → smaller area → ES
    y1_es = [70, 60, 50, 40, 30, 20]
    x2_es = [70, 71, 70, 66, 60, 56]
    y2_es = [70, 60, 50, 40, 30, 20]

    rows_001 = {
        "FileName": ["MOCK_001.avi"] * 12,
        "X1": x1_ed + x1_es,
        "Y1": y1_ed + y1_es,
        "X2": x2_ed + x2_es,
        "Y2": y2_ed + y2_es,
        "Frame": [10] * 6 + [25] * 6,
    }
    rows_002 = {
        "FileName": ["MOCK_002.avi"] * 12,
        "X1": x1_ed + x1_es,
        "Y1": y1_ed + y1_es,
        "X2": x2_ed + x2_es,
        "Y2": y2_ed + y2_es,
        "Frame": [12] * 6 + [27] * 6,
    }
    import pandas as _pd
    df_tracings = _pd.concat([_pd.DataFrame(rows_001), _pd.DataFrame(rows_002)], ignore_index=True)
    df_tracings.to_csv(f"{test_dir}/VolumeTracings.csv", index=False)
    print(" -> Mock CSVs generated.")

    # 3. Test Parser
    print("[3/5] Testing EchoNetParser...")
    parser = EchoNetParser(f"{test_dir}/FileList.csv", f"{test_dir}/VolumeTracings.csv")
    train_df = parser.get_split("TRAIN")
    assert len(train_df) == 1, f"Expected 1 train item, got {len(train_df)}"
    ed_f, es_f = parser.get_cardiac_phase_frames("MOCK_001.avi")
    # Frame 10 uses wider coords → larger LV area → ED; Frame 25 uses narrower → ES
    assert ed_f == 10 and es_f == 25, f"Expected ED=10, ES=25, got ED={ed_f}, ES={es_f}"
    print(f" -> Parser verified: cardiac phases ED={ed_f}, ES={es_f} (area-based).")

    # 4. Test Dataset and DataLoader
    print("[4/5] Testing PyTorch EchoNetVideoDataset & DataLoader...")
    dataset = EchoNetVideoDataset(
        df=train_df,
        videos_dir=f"{test_dir}/Videos",
        num_frames=32,
        image_size=112,
        is_train=True,
        parser=parser,
        return_phase_mask=True
    )
    assert len(dataset) == 1, "Dataset length mismatch"
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    batch = next(iter(loader))
    video_tensor = batch["video"]
    ef_target = batch["ef"]
    ed_mask = batch["ed_mask"]

    print(f" -> Batch Video Tensor Shape: {video_tensor.shape} (Expected: [1, 3, 32, 112, 112])")
    print(f" -> Batch EF Target: {ef_target} (Expected: 58.5)")
    print(f" -> Batch ED Mask Shape: {ed_mask.shape} (Expected: [1, 112, 112])")

    assert video_tensor.shape == (1, 3, 32, 112, 112), f"Video shape mismatch: {video_tensor.shape}"
    assert torch.isclose(ef_target, torch.tensor([58.5])), "Target EF mismatch"
    assert ed_mask.shape == (1, 112, 112), f"Mask shape mismatch: {ed_mask.shape}"

    # 5. Clean up temporary test data
    shutil.rmtree(test_dir)
    print("[5/5] Cleanup completed.")
    print("=" * 60)
    print("SUCCESS: ALL PHASE 1 DATA PIPELINE CHECKS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline_smoketest()
