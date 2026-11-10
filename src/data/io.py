"""
EchoNet-Dynamic Video I/O
==========================
Provides efficient, research-grade video decoding optimised for compressed
AVI files (H.264 / MJPEG) as found in the EchoNet-Dynamic dataset.
"""
import os
from typing import Optional

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# EchoNet-Dynamic channel statistics ([0,1] domain)
_ECHO_MEAN: float = 0.12
_ECHO_STD: float = 0.18


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_ffmpeg():
    if "IMAGEIO_FFMPEG_EXE" not in os.environ:
        local_ffmpeg = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            ".venv", "Scripts", "ffmpeg.exe"
        )
        if os.path.exists(local_ffmpeg):
            os.environ["IMAGEIO_FFMPEG_EXE"] = local_ffmpeg


def _imageio_load_clip(
    video_path: str,
    start_frame: int,
    num_frames: int,
    stride: int,
    target_size: int,
    mean: float,
    std: float,
) -> np.ndarray:
    """
    Decodes video frames using imageio (uses bundled static ffmpeg.exe).
    Supports MP4, AVI, and MOV seamlessly on Windows ARM64 and Linux.
    """
    from PIL import Image
    import imageio.v3 as iio

    _ensure_ffmpeg()
    raw = iio.imread(video_path)  # (T, H, W, C) uint8
    total = len(raw)

    frames_out: list = []
    last_valid: Optional[np.ndarray] = None

    for i in range(num_frames):
        raw_idx = start_frame + i * stride
        if raw_idx < total:
            frame = raw[raw_idx]
            if frame.shape[0] != target_size or frame.shape[1] != target_size:
                frame = np.array(
                    Image.fromarray(frame).resize(
                        (target_size, target_size), Image.Resampling.BILINEAR
                    ),
                    dtype=np.uint8,
                )
            last_valid = frame
        else:
            frame = last_valid if last_valid is not None else np.zeros(
                (target_size, target_size, 3), dtype=np.uint8
            )
        frames_out.append(frame)

    clip = np.stack(frames_out, axis=0).astype(np.float32) / 255.0  # [0,1]
    clip = (clip - mean) / std
    return clip


def _load_npy_clip(
    npy_path: str,
    start_frame: int,
    num_frames: int,
    stride: int,
    target_size: int,
    mean: float,
    std: float,
) -> np.ndarray:
    """
    Load a clip from a companion .npy file (used for local CPU testing on
    Windows ARM64 where cv2 may lack AVI codec support).
    """
    from PIL import Image

    raw = np.load(npy_path)  # (T, H, W, C) uint8
    total = len(raw)

    frames_out: list = []
    last_valid: Optional[np.ndarray] = None

    for i in range(num_frames):
        raw_idx = start_frame + i * stride
        if raw_idx < total:
            frame = raw[raw_idx]
            if frame.shape[0] != target_size or frame.shape[1] != target_size:
                frame = np.array(
                    Image.fromarray(frame).resize(
                        (target_size, target_size), Image.Resampling.BILINEAR
                    ),
                    dtype=np.uint8,
                )
            last_valid = frame
        else:
            frame = last_valid if last_valid is not None else np.zeros(
                (target_size, target_size, 3), dtype=np.uint8
            )
        frames_out.append(frame)

    clip = np.stack(frames_out, axis=0).astype(np.float32) / 255.0  # [0,1]
    clip = (clip - mean) / std
    return clip  # (num_frames, H, W, 3)


def _cv2_load_clip(
    video_path: str,
    start_frame: int,
    num_frames: int,
    stride: int,
    target_size: int,
    mean: float,
    std: float,
) -> np.ndarray:
    """
    Single-pass bounded reader using grab/retrieve (no seek-thrashing).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"cv2.VideoCapture could not open: {video_path}\n"
            "Ensure the file is a valid, non-corrupted AVI / MP4."
        )

    for _ in range(start_frame):
        if not cap.grab():
            break

    frames_out: list = []
    last_valid: Optional[np.ndarray] = None
    absolute_pos: int = start_frame

    while len(frames_out) < num_frames:
        relative_pos = absolute_pos - start_frame
        need_retrieve = (relative_pos % stride == 0)
        ok = cap.grab()

        if need_retrieve:
            if ok:
                ret, frame_bgr = cap.retrieve()
                if ret and frame_bgr is not None:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    if frame_rgb.shape[0] != target_size or frame_rgb.shape[1] != target_size:
                        frame_rgb = cv2.resize(
                            frame_rgb,
                            (target_size, target_size),
                            interpolation=cv2.INTER_AREA,
                        )
                    last_valid = frame_rgb
                    frames_out.append(frame_rgb)
                else:
                    _append_padding(frames_out, last_valid, target_size)
            else:
                while len(frames_out) < num_frames:
                    _append_padding(frames_out, last_valid, target_size)
                break

        absolute_pos += 1

    cap.release()

    clip = np.stack(frames_out, axis=0).astype(np.float32) / 255.0  # [0,1]
    clip = (clip - mean) / std
    return clip  # (num_frames, H, W, 3)


def _append_padding(
    frames: list,
    last_valid: Optional[np.ndarray],
    target_size: int,
) -> None:
    if last_valid is not None:
        frames.append(last_valid.copy())
    else:
        frames.append(np.zeros((target_size, target_size, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_video_clip(
    video_path: str,
    start_frame: int,
    num_frames: int = 32,
    stride: int = 2,
    target_size: int = 112,
    mean: float = _ECHO_MEAN,
    std: float = _ECHO_STD,
) -> np.ndarray:
    """
    Load a single normalised clip from an echocardiography video.
    """
    base_stem = os.path.splitext(video_path)[0]
    npy_path = video_path if video_path.endswith(".npy") else f"{base_stem}.npy"
    if os.path.exists(npy_path):
        return _load_npy_clip(npy_path, start_frame, num_frames, stride, target_size, mean, std)

    # Case-insensitive filesystem fallback if path not directly found
    if not os.path.exists(video_path):
        dir_name = os.path.dirname(video_path)
        base_name = os.path.basename(video_path).lower()
        if os.path.exists(dir_name):
            try:
                for entry in os.scandir(dir_name):
                    if entry.name.lower() == base_name:
                        video_path = entry.path
                        break
            except Exception:
                pass

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video file not found: {video_path}\n"
            "Ensure the video exists on disk or companion .npy is provided."
        )

    if _HAS_CV2:
        return _cv2_load_clip(video_path, start_frame, num_frames, stride, target_size, mean, std)
    return _imageio_load_clip(video_path, start_frame, num_frames, stride, target_size, mean, std)


def get_video_total_frames(video_path: str) -> int:
    """
    Return the number of decodable frames in a video.
    """
    base_stem = os.path.splitext(video_path)[0]
    npy_path = video_path if video_path.endswith(".npy") else f"{base_stem}.npy"
    if os.path.exists(npy_path):
        return len(np.load(npy_path, mmap_mode="r"))

    if not os.path.exists(video_path):
        dir_name = os.path.dirname(video_path)
        base_name = os.path.basename(video_path).lower()
        if os.path.exists(dir_name):
            try:
                for entry in os.scandir(dir_name):
                    if entry.name.lower() == base_name:
                        video_path = entry.path
                        break
            except Exception:
                pass

    if not os.path.exists(video_path):
        return 0

    if _HAS_CV2:
        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(n, 0)
    else:
        _ensure_ffmpeg()
        try:
            import imageio.v3 as iio
            props = iio.improps(video_path)
            if hasattr(props, "n_images") and props.n_images:
                return int(props.n_images)
            raw = iio.imread(video_path)
            return len(raw)
        except Exception:
            return 0


def load_multiclip_video(
    video_path: str,
    num_clips: int = 4,
    clip_len: int = 32,
    stride: int = 2,
    target_size: int = 112,
    mean: float = _ECHO_MEAN,
    std: float = _ECHO_STD,
) -> np.ndarray:
    """
    Extract `num_clips` evenly-spaced normalised clips from a video for
    beat-to-beat test-time averaging (multi-clip inference).
    """
    total = get_video_total_frames(video_path)
    span = clip_len * stride

    if total <= span or num_clips == 1:
        start_frames = [0] * num_clips
    else:
        max_start = total - span
        start_frames = [
            int(round(s))
            for s in np.linspace(0, max_start, num=num_clips)
        ]

    clips = [
        load_video_clip(
            video_path,
            start_frame=sf,
            num_frames=clip_len,
            stride=stride,
            target_size=target_size,
            mean=mean,
            std=std,
        )
        for sf in start_frames
    ]
    return np.stack(clips, axis=0)  # (num_clips, clip_len, H, W, 3)
