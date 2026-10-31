import os
import numpy as np
from PIL import Image, ImageDraw


def generate_sample_echo_videos(output_dir: str = "./demo_samples"):
    """
    Generate three realistic synthetic A4C echocardiogram clips
    representing the three main clinical categories:
    1. Normal EF (~62%) - vigorous radial wall motion
    2. Mildly Reduced EF (~46%) - attenuated excursion
    3. Severely Reduced EF (~28%) - severe hypokinesis
    """
    os.makedirs(output_dir, exist_ok=True)

    samples = [
        {"filename": "normal_lvef_62.avi", "base_r": (22, 34), "contract_pct": 0.35, "ef": 62.0},
        {"filename": "mild_lvef_46.avi", "base_r": (24, 35), "contract_pct": 0.20, "ef": 46.0},
        {"filename": "severe_lvef_28.avi", "base_r": (26, 37), "contract_pct": 0.08, "ef": 28.0},
    ]

    for sample in samples:
        avi_path = os.path.join(output_dir, sample["filename"])
        npy_path = avi_path.replace(".avi", ".npy")
        if os.path.exists(npy_path):
            continue

        frames = []
        rx_base, ry_base = sample["base_r"]
        contract_pct = sample["contract_pct"]

        for f in range(40):
            # Ultrasound sector fan background
            img = Image.new("RGB", (112, 112), (12, 12, 16))
            draw = ImageDraw.Draw(img)

            # Draw acoustic ultrasound sector beam (dim fan)
            sector_points = [(56, 10), (10, 105), (102, 105)]
            draw.polygon(sector_points, fill=(20, 24, 28))

            # Pulsating Left Ventricle cavity (sinusoidal heartbeat cycle)
            phase = 2 * np.pi * f / 28.0
            scale = 1.0 - contract_pct * (0.5 + 0.5 * np.sin(phase))

            rx = int(rx_base * scale)
            ry = int(ry_base * scale)
            cx, cy = 56, 58

            # Myocardial wall (muscle border)
            draw.ellipse([cx - rx - 3, cy - ry - 4, cx + rx + 3, cy + ry + 4], fill=(70, 75, 80))
            # Blood pool (dark cavity)
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(15, 18, 22))

            # Mitral valve leaflets fluttering
            valve_open = 0.5 + 0.5 * np.sin(phase + np.pi / 4)
            if valve_open > 0.4:
                draw.line([(cx - 8, cy + ry - 8), (cx - 2, cy + ry - 4)], fill=(180, 180, 180), width=1)
                draw.line([(cx + 8, cy + ry - 8), (cx + 2, cy + ry - 4)], fill=(180, 180, 180), width=1)

            # Subtle ultrasound speckle texture
            frame_arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, 4.0, frame_arr.shape)
            frame_arr = np.clip(frame_arr + noise, 0, 255).astype(np.uint8)

            frames.append(frame_arr)

        frames_np = np.array(frames, dtype=np.uint8)
        np.save(npy_path, frames_np)
        with open(avi_path, "wb") as f:
            f.write(b"RIFF_MOCK_ECHO")

    print(f"[DemoSamples] Preset echocardiography samples ready in: {output_dir}")

