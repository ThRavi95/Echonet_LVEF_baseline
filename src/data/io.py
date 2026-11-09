"""
EchoNet-Dynamic Video I/O
==========================
Provides efficient, research-grade video decoding optimised for compressed
AVI files (H.264 / MJPEG) as found in the EchoNet-Dynamic dataset.

Design principles:
  • Single-pass bounded read: cap.grab() advances the internal demuxer
    position without decoding.  cap.retrieve() is called only for the
    frames that are actually needed.  This eliminates keyframe seek
    thrashing that results from calling cap.set(CAP_PROP_POS_FRAMES, idx)
    inside a loop.
  • Short-video padding: if the video ends before `num_frames` are
    accumulated the last decoded frame is repeated to fill the gap.
  • Normalisation: frames are cast to float32, scaled to [0.0, 1.0],
    then standardised with the EchoNet-Dynamic channel statistics.
  • Returns NumPy arrays (not tensors); the Dataset layer handles the
    final PyTorch conversion and spatial augmentation.
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

    Returns float32 array of shape (num_frames, target_size, target_size, 3)
    normalised to zero-mean / unit-std.
    """
    from PIL import Image  # local import to keep cv2-free paths clean

    raw = np.load(npy_path)  # (T, H, W, C) uint8
    total = len(raw)

    frames_out: list = []
    last_valid: Optional[np.ndarray] = None

    for i in range(num_frames):
        raw_idx = start_frame + i * stride
        if raw_idx < total:
            frame = raw[raw_idx]
            # Resize if necessary
            if frame.shape[0] != target_size or frame.shape[1] != target_size:
                frame = np.array(
                    Image.fromarray(frame).resize(
                        (target_size, target_size), Image.Resampling.BILINEAR
                    ),
                    dtype=np.uint8,
                )
            last_valid = frame
        else:
            # Pad with last valid frame
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

    Algorithm:
      1. Fast-forward to start_frame using consecutive cap.grab() calls
         (advances demuxer without decoding).
      2. Iterate frame-by-frame; call cap.retrieve() only for every
         stride-th position to accumulate num_frames decoded frames.
      3. Pad the tail with the last valid frame if the video is too short.

    Returns float32 array (num_frames, target_size, target_size, 3) in [μ-normalised].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"cv2.VideoCapture could not open: {video_path}\n"
            "Ensure the file is a valid, non-corrupted AVI / MP4."
        )

    # --- Fast-forward to start_frame using grab() (no decode) ---
    for _ in range(start_frame):
        if not cap.grab():
            break  # video shorter than start_frame; clip will be all-padded

    frames_out: list = []
    last_valid: Optional[np.ndarray] = None
    absolute_pos: int = start_frame  # current position in the raw video timeline

    while len(frames_out) < num_frames:
        relative_pos = absolute_pos - start_frame  # position within the read window

        need_retrieve = (relative_pos % stride == 0)
        ok = cap.grab()

        if need_retrieve:
            if ok:
                # Only decode frames we actually need
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
                    # Decode failed: pad with last valid
                    _append_padding(frames_out, last_valid, target_size)
            else:
                # End of stream: pad remainder
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
    """Append the last valid frame (or a black frame) to `frames` as padding."""
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

    The function selects `num_frames` frames starting at `start_frame`
    with an inter-frame stride of `stride` raw frames.  The effective
    temporal window is `num_frames * stride` raw frames.

    Args:
        video_path:  Path to the .avi file (or .npy companion for local tests).
        start_frame: Index of the first raw frame to include.
        num_frames:  Number of output frames (temporal depth, default 32).
        stride:      Raw-frame stride between consecutive output frames
                     (default 2, matching Ouyang et al. Nature 2020).
        target_size: Spatial H=W after resize (default 112).
        mean:        Per-channel mean for normalisation ([0,1] domain).
        std:         Per-channel std  for normalisation ([0,1] domain).

    Returns:
        np.ndarray of shape (num_frames, target_size, target_size, 3),
        dtype float32, normalised to (pixel/255 - mean) / std.
    """
    # Companion .npy takes priority (local CPU development / testing)
    npy_path = video_path if video_path.endswith(".npy") else video_path.replace(".avi", ".npy")
    if os.path.exists(npy_path):
        return _load_npy_clip(npy_path, start_frame, num_frames, stride, target_size, mean, std)

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video file not found: {video_path}\n"
            "On Windows ARM64 without AVI codec support, place a companion .npy "
            "file (same basename) alongside the .avi for local CPU testing."
        )

    if not _HAS_CV2:
        raise RuntimeError(
            "OpenCV (cv2) is required for AVI decoding but is not installed.\n"
            "On Kaggle/Linux: cv2 is available natively.\n"
            "On local machines: provide a companion .npy file for CPU tests."
        )

    return _cv2_load_clip(video_path, start_frame, num_frames, stride, target_size, mean, std)


def get_video_total_frames(video_path: str) -> int:
    """
    Return the number of decodable frames in a video.

    For .npy companions this is len(array).  For AVI files this uses
    cv2.CAP_PROP_FRAME_COUNT (which may be approximate for some codecs;
    we always clamp start_frame inside load_video_clip so this is safe).

    Args:
        video_path: Path to .avi or .npy.

    Returns:
        Integer frame count (≥ 0).
    """
    npy_path = video_path if video_path.endswith(".npy") else video_path.replace(".avi", ".npy")
    if os.path.exists(npy_path):
        return len(np.load(npy_path, mmap_mode="r"))

    if not _HAS_CV2 or not os.path.exists(video_path):
        return 0

    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 0)


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

    If the video is shorter than the full clip window, all clips start at 0
    (effectively repeating the same padded clip, which is consistent with
    the single-clip evaluation fall-through).

    Args:
        video_path: Path to .avi or companion .npy.
        num_clips:  Number of clips to extract (default 4).
        clip_len:   Temporal depth of each clip in output frames (default 32).
        stride:     Raw-frame stride (default 2).
        target_size: Spatial H=W of each frame (default 112).
        mean:       Normalisation mean ([0,1] domain).
        std:        Normalisation std  ([0,1] domain).

    Returns:
        np.ndarray of shape (num_clips, clip_len, target_size, target_size, 3),
        dtype float32, normalised.
    """
    total = get_video_total_frames(video_path)
    span = clip_len * stride  # raw frames consumed per clip

    # Determine evenly-spaced start positions across the video
    if total <= span or num_clips == 1:
        # All clips start at 0 (short video — will be padded internally)
        start_frames = [0] * num_clips
    else:
        max_start = total - span
        # linspace gives num_clips points in [0, max_start]
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
