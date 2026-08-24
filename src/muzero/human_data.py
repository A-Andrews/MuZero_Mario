"""Load converted human gameplay into a pinned replay buffer (imitation learning).

Reads the .npz segments produced by scripts/convert_human_bk2.py (filenames like
``sub-01_ses-001_task-mario_level-w1l1_rep-000_seg0.npz``) and reconstructs
`Trajectory` objects. The npz files carry extra metadata keys (``completed``,
``bk2``) that are not `Trajectory` fields, and store ``level``/``terminal`` as
0-d numpy scalars — both handled here.

The returned buffer is a plain `TrajectoryBuffer` sized exactly to its contents,
so nothing is ever evicted; with the converter's all-ones priorities and
alpha=1 the first pass over it is exactly uniform with IS weights == 1.

Subject and level filtering happen on the *filename* (both are encoded in the
bk2 stem), so filtered-out files are never decompressed — with the default
12-level training set that skips roughly half the 1.9 GB corpus.
"""
from __future__ import annotations

import dataclasses
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.muzero.buffer import Trajectory, TrajectoryBuffer

_TRAJECTORY_FIELDS = frozenset(f.name for f in dataclasses.fields(Trajectory))
_LEVEL_RE = re.compile(r"^Level(\d+)-(\d+)$")


@dataclass
class HumanEvalSet:
    """Held-out human (obs, action) pairs for BC accuracy / cross-entropy eval."""

    obs: np.ndarray      # (N, C, H, W) uint8
    actions: np.ndarray  # (N,) int64

    def __len__(self) -> int:
        return int(self.actions.shape[0])


def load_trajectory_npz(path: Path) -> Tuple[Trajectory, bool]:
    """Load one converted segment -> (Trajectory, completed).

    Filters the npz keys down to the `Trajectory` dataclass fields and casts
    the 0-d ``level``/``terminal`` scalars to plain str/bool.
    """
    with np.load(path, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files if k in _TRAJECTORY_FIELDS}
        completed = bool(d["completed"]) if "completed" in d.files else False
    if "level" in data:
        data["level"] = str(data["level"])
    if "terminal" in data:
        data["terminal"] = bool(data["terminal"])
    return Trajectory(**data), completed


def level_filename_tag(level: str) -> str:
    """Map an env.levels name (``Level1-1``) to its bk2 filename tag (``level-w1l1``)."""
    m = _LEVEL_RE.match(level)
    if m is None:
        raise ValueError(f"unrecognised level name {level!r} (expected e.g. 'Level1-1')")
    return f"level-w{m.group(1)}l{m.group(2)}"


def select_human_files(
    data_dir: Path,
    subjects: Optional[Sequence[str]] = None,
    levels: Optional[Sequence[str]] = None,
) -> List[Path]:
    """List converted segments filtered by subject prefix and level tag."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("sub-*.npz"))
    if not files:
        raise FileNotFoundError(f"no converted human trajectories (sub-*.npz) in {data_dir}")

    if subjects is not None:
        available = sorted({f.name.split("_", 1)[0] for f in files})
        unknown = sorted(set(subjects) - set(available))
        if unknown:
            raise ValueError(f"unknown subject(s) {unknown}; available: {available}")
        prefixes = tuple(f"{s}_" for s in subjects)
        files = [f for f in files if f.name.startswith(prefixes)]

    if levels is not None:
        tags = {f"_{level_filename_tag(lv)}_" for lv in levels}
        files = [f for f in files if any(tag in f.name for tag in tags)]

    return files


def split_human_files(
    data_dir: Path,
    *,
    subjects: Optional[Sequence[str]] = None,
    levels: Optional[Sequence[str]] = None,
    holdout_fraction: float = 0.0,
    seed: int = 0,
) -> Tuple[List[Path], List[Path]]:
    """Seeded file-level train/holdout split -> (train_files, holdout_files).

    Factored out so the BC/action-agreement evaluators can reproduce *exactly*
    the split `load_human_buffer` trained on and score only on held-out files.
    The shuffle is over the filtered list, so callers must pass the **same**
    ``levels``/``subjects``/``holdout_fraction``/``seed`` to get the same split —
    filtering to one level afterwards is fine, re-splitting per level is not.
    """
    files = select_human_files(data_dir, subjects=subjects, levels=levels)
    rng = np.random.default_rng(seed)
    files = [files[i] for i in rng.permutation(len(files))]
    n_holdout = int(round(len(files) * float(holdout_fraction)))
    return files[n_holdout:], files[:n_holdout]


def load_human_buffer(
    data_dir: str | Path,
    *,
    unroll_K: int,
    num_actions: int,
    levels: Optional[Sequence[str]] = None,
    subjects: Optional[Sequence[str]] = None,
    completed_only: bool = False,
    max_transitions: int = 0,
    holdout_fraction: float = 0.0,
    holdout_max_transitions: int = 4096,
    buffer_kwargs: Optional[dict] = None,
    seed: int = 0,
    num_threads: int = 8,
) -> Tuple[TrajectoryBuffer, Optional[HumanEvalSet]]:
    """Build a pinned TrajectoryBuffer (plus optional BC-eval holdout) from the corpus.

    ``levels=None`` keeps all levels; ``subjects=None`` keeps all subjects.
    ``max_transitions>0`` caps the load at whole-trajectory granularity after a
    seeded shuffle, so the retained subset is an unbiased sample of the corpus.
    The holdout is split at *file* level before the cap, so eval pairs are
    never also trained on.
    """
    t0 = time.time()
    train_files, holdout_files = split_human_files(
        Path(data_dir),
        subjects=subjects,
        levels=levels,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )
    if not train_files and not holdout_files:
        raise ValueError(
            f"no human trajectories left after filtering "
            f"(subjects={subjects}, levels={levels}) in {data_dir}"
        )

    pool = ThreadPoolExecutor(max_workers=max(1, num_threads))
    try:
        # --- holdout eval set --------------------------------------------------
        eval_set: Optional[HumanEvalSet] = None
        if holdout_files and holdout_max_transitions > 0:
            obs_parts: List[np.ndarray] = []
            act_parts: List[np.ndarray] = []
            n_eval = 0
            for traj, completed in pool.map(load_trajectory_npz, holdout_files):
                if completed_only and not completed:
                    continue
                obs_parts.append(traj.obs_stacks)
                act_parts.append(traj.actions)
                n_eval += traj.length
                if n_eval >= holdout_max_transitions:
                    break
            if obs_parts:
                eval_set = HumanEvalSet(
                    obs=np.concatenate(obs_parts, axis=0)[:holdout_max_transitions],
                    actions=np.concatenate(act_parts, axis=0)[:holdout_max_transitions],
                )

        # --- training trajectories --------------------------------------------
        trajectories: List[Trajectory] = []
        total = 0
        completions = 0
        per_level: Dict[str, int] = {}
        chunk = max(1, num_threads) * 8
        for start in range(0, len(train_files), chunk):
            if max_transitions > 0 and total >= max_transitions:
                break
            for traj, completed in pool.map(load_trajectory_npz, train_files[start:start + chunk]):
                if completed_only and not completed:
                    continue
                if max_transitions > 0 and total >= max_transitions:
                    break
                assert traj.policies.shape[1] == num_actions, (
                    f"human policies have {traj.policies.shape[1]} actions, model expects {num_actions}"
                )
                trajectories.append(traj)
                total += traj.length
                completions += int(completed)
                per_level[traj.level] = per_level.get(traj.level, 0) + traj.length
    finally:
        pool.shutdown(wait=False)

    if total == 0:
        raise ValueError(
            f"0 human transitions survived filtering "
            f"(subjects={subjects}, levels={levels}, completed_only={completed_only})"
        )

    buf = TrajectoryBuffer(
        capacity_transitions=total,  # sized to contents -> pinned, never evicts
        unroll_K=unroll_K,
        num_actions=num_actions,
        **(buffer_kwargs or {}),
    )
    for traj in trajectories:
        buf.add(traj, rebuild=False)
    buf._rebuild_flat()

    level_summary = ", ".join(f"{lv}={n}" for lv, n in sorted(per_level.items()))
    print(
        f"[human] loaded {len(trajectories)} segments / {total} transitions "
        f"({completions} completed) from {len(train_files)} files in {time.time() - t0:.1f}s; "
        f"holdout={len(eval_set) if eval_set else 0} transitions; per-level: {level_summary}",
        flush=True,
    )
    return buf, eval_set
