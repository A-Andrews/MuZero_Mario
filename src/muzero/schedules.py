"""Piecewise-linear schedules keyed on training steps.

A schedule is a list of ``(train_step, value)`` knots with strictly ascending
steps. Between knots the value is linearly interpolated; outside the range it
clamps to the first/last value. Knot steps return their stored value exactly,
with no float interpolation residue.

Shared by the imitation mix ratio (`MixedBuffer`) and the root-Dirichlet
exploration anneal (`src/selfplay/worker.py`). Note this is *not* the same
shape as `temperature.py`, which is deliberately piecewise-**constant**:
temperature steps down at thresholds, these anneal smoothly.
"""
from typing import List, Optional, Sequence, Tuple

Schedule = List[Tuple[int, float]]


def validate_schedule(
    schedule: Optional[Sequence],
    lo: Optional[float] = None,
    hi: Optional[float] = None,
) -> Optional[Schedule]:
    """Normalise to `(int, float)` knots. None/empty returns None."""
    if not schedule:
        return None
    pts = [(int(s), float(v)) for s, v in schedule]
    if lo is not None or hi is not None:
        assert all(
            (lo is None or v >= lo) and (hi is None or v <= hi) for _, v in pts
        ), f"schedule values must be in [{lo}, {hi}]: {pts}"
    assert all(
        pts[i][0] < pts[i + 1][0] for i in range(len(pts) - 1)
    ), f"schedule steps must be strictly ascending: {pts}"
    return pts


def value_at(step: int, points: Schedule) -> float:
    """Interpolated value at `step`, clamped outside the knot range."""
    if step <= points[0][0]:
        return points[0][1]
    for (s0, v0), (s1, v1) in zip(points, points[1:]):
        # Strict < so knot steps fall through and return their stored value
        # exactly (no float interpolation residue).
        if step < s1:
            return v0 + (v1 - v0) * (step - s0) / (s1 - s0)
    return points[-1][1]


def value_at_gated(step: int, points: Schedule, origin: Optional[int]) -> float:
    """`value_at` with the schedule's clock started at `origin`.

    ``origin=None`` means the gate has not opened yet, and pins the value to the
    first knot however far `step` has advanced. Used by the completion-gated
    root-Dirichlet anneal (`mcts.root_exploration_eps_gate_on_completion`): a
    run that has never completed a level keeps its full exploration budget
    rather than annealing into a local optimum it can no longer escape.
    """
    if origin is None:
        return points[0][1]
    return value_at(step - origin, points)
