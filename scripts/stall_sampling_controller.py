"""Causal progress monitor for evaluation-only, stall-triggered MCTS sampling."""
from collections import deque

SETTINGS = {"window_decisions": 96, "max_x_span": 16, "burst_decisions": 32,
            "sampling_temperature": .25, "normal_player_state": 8}


class StallSamplingController:
    """Read current state before acting; never consume randomness or future data.

    A window of W decisions needs W+1 pre-action positions. After the fixed
    sampling burst, require W new greedy decisions before another trigger.
    Non-control states clear history and cancel sampling.
    """
    def __init__(self, window_decisions=96, max_x_span=16, burst_decisions=32,
                 sampling_temperature=.25, normal_player_state=8):
        if window_decisions < 1 or burst_decisions < 1 or max_x_span < 0 or sampling_temperature <= 0:
            raise ValueError("Invalid stall-controller settings")
        self.positions = deque(maxlen=window_decisions+1)
        self.window_decisions = window_decisions
        self.max_x_span = max_x_span
        self.burst_decisions = burst_decisions
        self.sampling_temperature = sampling_temperature
        self.normal_player_state = normal_player_state
        self.remaining = 0

    def observe(self, x, player_state):
        trigger, span = False, -1
        if player_state != self.normal_player_state:
            self.positions.clear()
            self.remaining = 0
        elif self.remaining == 0:
            self.positions.append(int(x))
            if len(self.positions) == self.window_decisions+1:
                span = max(self.positions)-min(self.positions)
                if span <= self.max_x_span:
                    trigger = True
                    self.remaining = self.burst_decisions
        active = self.remaining > 0
        if active:
            self.remaining -= 1
            if self.remaining == 0:
                self.positions.clear()
        return {"controller_temperature": self.sampling_temperature if active else 0.,
                "rescue_active": active, "rescue_trigger": trigger,
                "rescue_remaining_after_action": self.remaining, "stall_window_span": span}
