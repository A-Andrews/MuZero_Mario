"""Write rollout frames to mp4 via imageio (imageio-ffmpeg's bundled ffmpeg,
so no system ffmpeg module is needed on the cluster).

NOTE: this module was reconstructed from its call sites after the original
(untracked on BMRC — a bare `logs/` .gitignore pattern also matched `src/logs/`)
was lost in the Isambard migration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import numpy as np


def save_video(path: Path | str, frames: Sequence[np.ndarray], fps: int = 15):
    """Encode a list of (H, W, 3) uint8 RGB frames as an mp4.

    ffmpeg requires even dimensions for yuv420p; odd-sized frames are padded
    by one row/column (macro_block_size=1 would break some players).
    """
    if not frames:
        raise ValueError(f"no frames to write to {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    stacked = [np.asarray(f, dtype=np.uint8) for f in frames]
    h, w = stacked[0].shape[:2]
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        stacked = [
            np.pad(f, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge") for f in stacked
        ]

    with imageio.get_writer(
        str(path), fps=fps, codec="libx264", quality=7, macro_block_size=1
    ) as writer:
        for frame in stacked:
            writer.append_data(frame)
    return path
