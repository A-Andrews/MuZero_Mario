import cv2
import numpy as np


def grayscale_resize(frame):
    """Convert a raw RGB frame to a 84x84 float32 grayscale image in [0, 1]."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (84, 84))
    return gray.astype(np.float32) / 255.0


def stack_max_pooled(gray_stack, n_frame, downsample, pad_to=None):
    """Build the MuZero observation tensor from an already-grayscaled deque.

    `gray_stack` is an iterable of float32 84x84 frames of length
    n_frame * downsample. We take the max of two consecutive frames per
    downsample window (flicker removal), producing n_frame output channels.

    Optionally pad each channel to (pad_to, pad_to) by edge padding so that
    repeated stride-2 convs divide cleanly (e.g., 96 for MuZero-Atari).

    Returns an ndarray of shape (n_frame, H, W) float32.
    """
    frames = list(gray_stack)
    out_frames = []
    for i in range(n_frame - 1, -1, -1):
        a = frames[-i * downsample - 2]
        b = frames[-i * downsample - 1]
        out_frames.append(np.maximum(a, b))
    stacked = np.stack(out_frames, axis=0).astype(np.float32)
    if pad_to is not None and pad_to != stacked.shape[-1]:
        side = stacked.shape[-1]
        pad = (pad_to - side) // 2
        stacked = np.pad(
            stacked,
            ((0, 0), (pad, pad_to - side - pad), (pad, pad_to - side - pad)),
            mode="edge",
        )
    return stacked


def preprocess_frames(frame_list, n_frame, downsample, pad_to=None):
    """Legacy API: accept raw RGB frames and apply the full pipeline.

    Kept for callers that still feed raw frames; the training path now caches
    grayscaled frames via `grayscale_resize` + `stack_max_pooled`.
    """
    gray = [grayscale_resize(f) for f in frame_list]
    return stack_max_pooled(gray, n_frame, downsample, pad_to=pad_to)
