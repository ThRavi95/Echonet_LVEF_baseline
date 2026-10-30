import os
import numpy as np
from PIL import Image

# Check for OpenCV availability, fallback to imageio
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

import imageio.v3 as iio


def load_video(video_path: str, target_size: int = 112) -> np.ndarray:
    """
    Load an echocardiography video (.avi) and resize frames.
    Supports native AVI decoding via OpenCV on Kaggle/Linux, and .npy mock/cached
    video arrays for local CPU development on Windows ARM64.
    
    Args:
        video_path: Path to .avi or .npy video file.
        target_size: Target square dimension (H=W=target_size).
        
    Returns:
        np.ndarray of shape (T, target_size, target_size, 3) with dtype uint8.
    """
    # Check for direct or companion .npy file (ideal for local tests & fast caching)
    npy_path = video_path if video_path.endswith(".npy") else video_path.replace(".avi", ".npy")
    if os.path.exists(npy_path):
        frames = np.load(npy_path)
        if frames.shape[1] != target_size or frames.shape[2] != target_size:
            resized_frames = []
            for f in frames:
                img = Image.fromarray(f).resize((target_size, target_size), Image.Resampling.BILINEAR)
                resized_frames.append(np.array(img))
            frames = np.array(resized_frames, dtype=np.uint8)
        return frames

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    frames = []

    if _HAS_CV2:
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_rgb.shape[0] != target_size or frame_rgb.shape[1] != target_size:
                frame_rgb = cv2.resize(frame_rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
            frames.append(frame_rgb)
        cap.release()
    else:
        # Fallback to imageio if available
        try:
            raw_frames = iio.imread(video_path)
            for frame in raw_frames:
                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                elif frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                    
                if frame.shape[0] != target_size or frame.shape[1] != target_size:
                    img = Image.fromarray(frame).resize((target_size, target_size), Image.Resampling.BILINEAR)
                    frame = np.array(img)
                frames.append(frame)
        except Exception as e:
            raise RuntimeError(
                f"Cannot decode video '{video_path}'. On Windows ARM64 without OpenCV/ffmpeg, "
                f"use .npy video arrays for local tests, or run on Kaggle GPU where OpenCV is native. Error: {e}"
            )

    if not frames:
        raise ValueError(f"Unable to decode frames from video: {video_path}")

    return np.array(frames, dtype=np.uint8)


def sample_video_clip(
    frames: np.ndarray,
    target_frames: int = 32,
    stride: int = 1,
    start_frame: int = None
) -> np.ndarray:
    """
    Extract a contiguous or strided sub-clip of length `target_frames`.
    If video has fewer frames than required, repeats the boundary frames.
    
    Args:
        frames: Array of shape (T, H, W, C).
        target_frames: Required temporal length (standard: 32).
        stride: Temporal stride between extracted frames.
        start_frame: Explicit starting frame. If None, centers the clip or starts at 0.
        
    Returns:
        np.ndarray of shape (target_frames, H, W, C).
    """
    total_frames = len(frames)
    span = target_frames * stride

    if total_frames < span:
        # Pad by repeating the last frame
        pad_count = span - total_frames
        padding = np.repeat(frames[-1:], pad_count, axis=0)
        frames = np.concatenate([frames, padding], axis=0)
        total_frames = len(frames)

    if start_frame is None:
        start_frame = max(0, (total_frames - span) // 2)
    else:
        start_frame = max(0, min(start_frame, total_frames - span))

    indices = [start_frame + i * stride for i in range(target_frames)]
    return frames[indices]
