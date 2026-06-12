"""Inverse-length, completion-aware autocurriculum for multi-level self-play.

The learner maintains a rolling window of recent episode lengths and
completion flags per level. Two signals combine multiplicatively:

  * inverse episode length — levels where Mario dies quickly (struggling)
    get sampled more;
  * incompletion rate — levels the agent already finishes reliably get
    sampled less (down to the ``min_weight`` floor, which guards against
    catastrophic forgetting), freeing worker time for unfinished levels.

Weights are written into a shared ``mp.Array(c_double, num_levels)`` so workers
pick them up without IPC: at each episode boundary they snapshot the array and
sample a level. Until a level has any data the sampler returns a uniform
weight, so warm-up does not pin all workers to whichever level happens to
finish first.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Sequence


class LevelSampler:
    """Rolling per-level length tracker → inverse-length sampling weights.

    Parameters
    ----------
    levels:
        Ordered list of level names. Weight at index ``i`` corresponds to
        ``levels[i]``.
    history_size:
        Number of most-recent episode lengths kept per level for the running
        mean.
    exponent:
        Power applied to the inverse mean length. ``1.0`` gives strict inverse
        proportionality; values < 1 soften the curriculum, > 1 sharpen it.
    min_weight:
        Floor (after normalization) so no level is ever sampled with zero
        probability. Prevents the rare-but-catastrophic case where a level
        gets starved of fresh data and its mean length never updates.
    """

    def __init__(
        self,
        levels: Sequence[str],
        history_size: int = 50,
        exponent: float = 1.0,
        min_weight: float = 0.02,
    ):
        self.levels: List[str] = list(levels)
        self._idx: Dict[str, int] = {lv: i for i, lv in enumerate(self.levels)}
        self._history: Dict[str, deque] = {
            lv: deque(maxlen=int(history_size)) for lv in self.levels
        }
        self._completions: Dict[str, deque] = {
            lv: deque(maxlen=int(history_size)) for lv in self.levels
        }
        self.exponent = float(exponent)
        self.min_weight = float(min_weight)

    def record_episode(self, level: str, length: int, completed: bool = False) -> None:
        if level in self._history:
            self._history[level].append(int(length))
            self._completions[level].append(1.0 if completed else 0.0)

    def mean_lengths(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for lv in self.levels:
            h = self._history[lv]
            out[lv] = (sum(h) / len(h)) if h else float("nan")
        return out

    def completion_rates(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for lv in self.levels:
            c = self._completions[lv]
            out[lv] = (sum(c) / len(c)) if c else 0.0
        return out

    def compute_weights(self) -> Dict[str, float]:
        """Return a normalized weight per level (sums to 1.0)."""
        means = self.mean_lengths()
        completion = self.completion_rates()
        # Levels with no history get the average of the populated levels so
        # they are explored at a sensible baseline rate, not starved.
        populated = [v for v in means.values() if v == v]  # filter NaN
        fallback = (sum(populated) / len(populated)) if populated else 1.0
        raw = []
        for lv in self.levels:
            m = means[lv]
            if m != m:  # NaN
                m = fallback
            inv_len = 1.0 / max(float(m), 1.0) ** self.exponent
            # Down-weight mastered levels; the small offset keeps a fully
            # completed level from hitting exactly zero before the floor.
            incompletion = 1.0 - float(completion[lv]) + 0.05
            raw.append(inv_len * incompletion)
        total = sum(raw) or 1.0
        weights = [r / total for r in raw]
        # Apply the minimum-weight floor by pinning under-floor entries at the
        # floor and proportionally redistributing the *remaining* probability
        # mass among the rest. A naive max-then-renormalize would push floored
        # entries back below the floor, defeating the whole point.
        floor = max(0.0, float(self.min_weight))
        n = len(weights)
        if floor > 0.0 and floor * n < 1.0:
            below = [i for i, w in enumerate(weights) if w < floor]
            if below:
                budget = 1.0 - floor * len(below)
                above_total = sum(weights[i] for i in range(n) if i not in below)
                new_w = [0.0] * n
                for i in below:
                    new_w[i] = floor
                if above_total > 0.0:
                    for i in range(n):
                        if i not in below:
                            new_w[i] = weights[i] / above_total * budget
                else:
                    # All entries were sub-floor (degenerate); split budget evenly.
                    fill = budget / max(1, n - len(below))
                    for i in range(n):
                        if i not in below:
                            new_w[i] = fill
                weights = new_w
        elif floor > 0.0:
            # Floor too large to satisfy; degrade gracefully to uniform.
            weights = [1.0 / n] * n
        return {lv: w for lv, w in zip(self.levels, weights)}
