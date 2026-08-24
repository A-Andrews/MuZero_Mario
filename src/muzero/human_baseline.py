"""Per-level human reference statistics + held-out human actions.

The project's headline question is how the agent stacks up against a person on
the *same* level, so every checkpoint replay is scored against two human
references drawn from the converted CNeuroMod corpus
(``scripts/convert_human_bk2.py``, see ``outputs/human_trajectories/``):

* **Performance** — per-level completion rate, time-to-flag and return, from
  `HumanLevelStats`. Cached to ``<data_dir>/human_level_stats.json`` because
  scanning the corpus costs a few seconds and the answer never changes.
* **Behaviour** — held-out (obs, action) pairs per level, so the policy head's
  top-1 agreement with the human's actual button press can be measured on the
  states the human was in, not just the ones the agent reaches.

Two scale mismatches are handled here rather than at the call site:

* The converter baked in its own ``COMPLETION_BONUS`` (100 raw = +10 in agent
  reward units). A run overriding ``env.completion_bonus`` (the curriculum
  recipe uses 200) would otherwise be compared against a differently-scaled
  return. `HumanLevelStats.return_at_bonus` re-bases it — valid because the
  bonus is a single additive term on the completing step, and the replay
  metric is an *undiscounted* sum, so the converter's discount is irrelevant.
* Human reps are split at deaths into ``done_on_life_loss``-style segments, so
  a *rep* is one human attempt and a *segment* is one life. Completion rate is
  reported per rep (the human's success rate per attempt, comparable to a
  self-play episode) while lengths/returns come from the completing segment.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.muzero.human_data import (
    HumanEvalSet,
    level_filename_tag,
    load_trajectory_npz,
    split_human_files,
)

# Reward-shaping constants the corpus was converted under; mirrored from
# scripts/convert_human_bk2.py. Only the bonus needs rebasing (see module doc).
CONVERTER_COMPLETION_BONUS = 100.0
CONVERTER_REWARD_SCALE = 10.0  # rewards were stored as raw/10, as CustomWrapper does

STATS_CACHE_NAME = "human_level_stats.json"
_CACHE_VERSION = 3
_REP_RE = re.compile(r"_seg\d+$")


@dataclass
class HumanLevelStats:
    """Human reference for one level. Lengths are agent steps (frame_skip windows)."""

    level: str
    n_reps: int = 0
    n_segments: int = 0
    n_completed: int = 0          # reps with a completing segment
    n_no_death: int = 0           # reps completed on the first life
    completion_rate: float = 0.0  # n_completed / n_reps
    no_death_rate: float = 0.0
    length_mean: float = 0.0      # over completing segments
    length_median: float = 0.0
    length_min: float = 0.0
    length_p10: float = 0.0       # a "good human", not a median one
    length_p90: float = 0.0       # a slow-but-successful human
    return_mean: float = 0.0      # at CONVERTER_COMPLETION_BONUS
    return_median: float = 0.0
    subjects: List[str] = field(default_factory=list)

    @property
    def has_completions(self) -> bool:
        return self.n_completed > 0

    def return_at_bonus(self, completion_bonus: float, which: str = "median") -> float:
        """Median/mean completing return re-based to this run's completion bonus."""
        base = self.return_median if which == "median" else self.return_mean
        delta = (float(completion_bonus) - CONVERTER_COMPLETION_BONUS) / CONVERTER_REWARD_SCALE
        return base + delta


def _rep_key(path: Path) -> str:
    return _REP_RE.sub("", path.stem)


def _read_segment_meta(path: Path):
    """(steps, completed, undiscounted return) without decompressing obs_stacks."""
    with np.load(path, allow_pickle=False) as d:
        rewards = d["rewards"]
        completed = bool(d["completed"]) if "completed" in d.files else False
    return int(rewards.shape[0]), completed, float(rewards.sum())


def compute_level_stats(
    data_dir: str | Path,
    levels: Sequence[str],
    *,
    subjects: Optional[Sequence[str]] = None,
    use_cache: bool = True,
    num_threads: int = 8,
) -> Dict[str, HumanLevelStats]:
    """Per-level human reference stats, memoised in ``<data_dir>/human_level_stats.json``.

    Levels the humans never played (w2l2, w7l2, the castle -4s) simply come back
    absent from the dict — callers skip comparison for those.
    """
    data_dir = Path(data_dir)
    cache_path = data_dir / STATS_CACHE_NAME
    subj_key = ",".join(sorted(subjects)) if subjects else "all"

    cache: Dict[str, dict] = {}
    if use_cache and cache_path.is_file():
        try:
            blob = json.loads(cache_path.read_text())
            if blob.get("version") == _CACHE_VERSION:
                cache = blob.get("entries", {})
        except (OSError, ValueError) as e:
            print(f"[human-baseline] ignoring unreadable cache {cache_path}: {e}")

    out: Dict[str, HumanLevelStats] = {}
    missing = []
    for level in levels:
        entry = cache.get(f"{level}|{subj_key}")
        if entry is not None:
            out[level] = HumanLevelStats(**entry)
        else:
            missing.append(level)

    if missing:
        t0 = time.time()
        pool = ThreadPoolExecutor(max_workers=max(1, num_threads))
        try:
            for level in missing:
                stats = _scan_level(data_dir, level, subjects, pool)
                if stats is not None:
                    out[level] = stats
                    cache[f"{level}|{subj_key}"] = asdict(stats)
        finally:
            pool.shutdown(wait=False)
        try:
            cache_path.write_text(
                json.dumps({"version": _CACHE_VERSION, "entries": cache}, indent=1)
            )
        except OSError as e:  # a read-only corpus dir must not kill training
            print(f"[human-baseline] could not write {cache_path}: {e}")
        print(
            f"[human-baseline] scanned {len(missing)} level(s) in {time.time() - t0:.1f}s; "
            + ", ".join(
                f"{lv}={out[lv].completion_rate:.0%}/{out[lv].length_median:.0f}steps"
                for lv in missing if lv in out
            ),
            flush=True,
        )
    return out


def _scan_level(data_dir, level, subjects, pool) -> Optional[HumanLevelStats]:
    try:
        tag = level_filename_tag(level)
    except ValueError:
        return None
    files = sorted(Path(data_dir).glob(f"sub-*_{tag}_*.npz"))
    if subjects:
        files = [f for f in files if f.name.split("_", 1)[0] in set(subjects)]
    if not files:
        return None

    reps: Dict[str, list] = {}
    for path, meta in zip(files, pool.map(_read_segment_meta, files)):
        reps.setdefault(_rep_key(path), []).append(meta)

    lengths, returns, n_completed, n_no_death = [], [], 0, 0
    for segs in reps.values():
        if any(c for _, c, _ in segs):
            n_completed += 1
        if segs and segs[0][1]:
            n_no_death += 1
        for steps, completed, ret in segs:
            if completed:
                lengths.append(steps)
                returns.append(ret)

    n_reps = len(reps)
    return HumanLevelStats(
        level=level,
        n_reps=n_reps,
        n_segments=len(files),
        n_completed=n_completed,
        n_no_death=n_no_death,
        completion_rate=n_completed / n_reps if n_reps else 0.0,
        no_death_rate=n_no_death / n_reps if n_reps else 0.0,
        length_mean=float(np.mean(lengths)) if lengths else 0.0,
        length_median=float(np.median(lengths)) if lengths else 0.0,
        length_min=float(np.min(lengths)) if lengths else 0.0,
        length_p10=float(np.percentile(lengths, 10)) if lengths else 0.0,
        length_p90=float(np.percentile(lengths, 90)) if lengths else 0.0,
        return_mean=float(np.mean(returns)) if returns else 0.0,
        return_median=float(np.median(returns)) if returns else 0.0,
        subjects=sorted({f.name.split("_", 1)[0] for f in files}),
    )


def compare_metrics(
    level: str,
    *,
    completed: bool,
    n_steps: int,
    total_return: float,
    stats: Optional[HumanLevelStats],
    completion_bonus: float,
) -> Dict[str, float]:
    """Score one agent rollout against the human reference for the same level.

    Returns the `human/<level>_*` reference lines plus the `compare/<level>_*`
    ratios; empty when humans never played the level. The two sides are not
    measured identically and that is deliberate: the agent value is a single
    greedy, deterministic-start rollout while the human numbers are
    distributions over many attempts, so `completed_vs_human` reads as "did
    this model finish it" against "how often did a person".

    `length_ratio` is emitted only when the agent actually reached the flag —
    time-to-flag is meaningless on a run that died — and is the one metric here
    where **lower is better**.
    """
    if stats is None:
        return {}
    # Which human rate to compare against is not a free choice. Agent episodes
    # (and the greedy replay) run with `done_on_life_loss=True`, so one attempt
    # is *one life, from the level start*. The matching human number is
    # `no_death_rate` — reps whose FIRST segment reached the flag. The per-rep
    # rate counts a human who died twice and finished on the third life, which
    # the agent is never given the chance to do, and overstates the human by
    # 8-16 points on most levels.
    m: Dict[str, float] = {
        f"human/{level}_completion_rate_first_life": stats.no_death_rate,
        f"human/{level}_completion_rate_any_life": stats.completion_rate,
        f"compare/{level}_completed_vs_human": (
            (1.0 if completed else 0.0) - stats.no_death_rate
        ),
    }
    if not stats.has_completions:
        return m  # humans played it but never finished: no length/return reference
    human_return = stats.return_at_bonus(completion_bonus)
    m[f"human/{level}_length_median"] = stats.length_median
    m[f"human/{level}_length_p10"] = stats.length_p10
    m[f"human/{level}_return_median"] = human_return
    if human_return > 0:
        m[f"compare/{level}_return_ratio"] = total_return / human_return
    if completed and stats.length_median > 0:
        m[f"compare/{level}_length_ratio"] = n_steps / stats.length_median
        m[f"compare/{level}_beats_human_median"] = float(n_steps <= stats.length_median)
        m[f"compare/{level}_beats_human_p10"] = float(n_steps <= stats.length_p10)
    return m


def load_action_eval_sets(
    data_dir: str | Path,
    levels: Sequence[str],
    *,
    subjects: Optional[Sequence[str]] = None,
    max_transitions_per_level: int = 1024,
    holdout_only: bool = False,
    holdout_fraction: float = 0.0,
    split_levels: Optional[Sequence[str]] = None,
    seed: int = 0,
    num_threads: int = 8,
) -> Dict[str, HumanEvalSet]:
    """Per-level held-out human (obs, action) pairs for action-agreement scoring.

    ``holdout_only`` (set it whenever imitation training is on) restricts the
    draw to the *same* files `load_human_buffer` withheld. Reproducing that
    split requires the identical shuffle input, so ``split_levels`` must be the
    level list the imitation loader was given (``levels`` is only the subset we
    want metrics for); the split runs once and is bucketed by level afterwards.
    With imitation off nothing was trained on, so the whole corpus is fair game.

    RAM: obs are (4,96,96) uint8 = 36.9 KB/step, so the default is ~38 MB per
    level (~450 MB across the 12-level set).
    """
    if max_transitions_per_level <= 0:
        return {}
    data_dir = Path(data_dir)
    try:
        train_files, holdout_files = split_human_files(
            data_dir,
            subjects=subjects,
            levels=list(split_levels) if split_levels is not None else list(levels),
            holdout_fraction=holdout_fraction,
            seed=seed,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[human-baseline] no human corpus for action-agreement: {e}")
        return {}
    pool_files = holdout_files if holdout_only else (holdout_files + train_files)

    out: Dict[str, HumanEvalSet] = {}
    pool = ThreadPoolExecutor(max_workers=max(1, num_threads))
    try:
        for level in levels:
            try:
                tag = f"_{level_filename_tag(level)}_"
            except ValueError:
                continue
            files = [f for f in pool_files if tag in f.name]
            if not files:
                if holdout_only:
                    print(
                        f"[human-baseline] {level}: imitation holdout has no files "
                        f"(holdout_fraction={holdout_fraction}) — no action-agreement metric"
                    )
                continue

            obs_parts, act_parts, n = [], [], 0
            for traj, _completed in pool.map(load_trajectory_npz, files):
                obs_parts.append(traj.obs_stacks)
                act_parts.append(traj.actions)
                n += traj.length
                if n >= max_transitions_per_level:
                    break
            if not obs_parts:
                continue
            out[level] = HumanEvalSet(
                obs=np.concatenate(obs_parts, axis=0)[:max_transitions_per_level],
                actions=np.concatenate(act_parts, axis=0)[:max_transitions_per_level],
            )
            if len(out[level]) < 256:
                print(
                    f"[human-baseline] {level}: only {len(out[level])} eval transitions — "
                    f"action-agreement will be noisy"
                )
    finally:
        pool.shutdown(wait=False)

    if out:
        mb = sum(e.obs.nbytes for e in out.values()) / 1e6
        print(
            "[human-baseline] action-eval: "
            + ", ".join(f"{lv}={len(e)}" for lv, e in sorted(out.items()))
            + f" ({mb:.0f} MB, holdout_only={holdout_only})",
            flush=True,
        )
    return out
