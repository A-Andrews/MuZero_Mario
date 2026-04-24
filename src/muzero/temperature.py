"""Temperature schedule keyed on training steps.

`schedule` is a list of (step_threshold, temperature) pairs in ascending order;
the active temperature at a given step is the value of the largest threshold
not exceeding `step`.
"""
from typing import List, Tuple


def temperature_for_step(step: int, schedule: List[Tuple[int, float]]) -> float:
    assert schedule, "schedule must be non-empty"
    current = schedule[0][1]
    for threshold, t in schedule:
        if step >= threshold:
            current = t
        else:
            break
    return float(current)
