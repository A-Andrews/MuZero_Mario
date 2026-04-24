"""MinMaxStats for Q normalisation inside a search tree (ported from Muzero-Hanoi)."""


class MinMaxStats:
    def __init__(self, min_value_bound=None, max_value_bound=None):
        self.maximum = max_value_bound if max_value_bound is not None else -float("inf")
        self.minimum = min_value_bound if min_value_bound is not None else float("inf")

    def update(self, value):
        if value > self.maximum:
            self.maximum = value
        if value < self.minimum:
            self.minimum = value

    def normalize(self, value):
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
