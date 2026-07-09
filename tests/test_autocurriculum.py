"""Tests for the inverse-length level sampler."""
import math

import pytest

from src.selfplay.autocurriculum import LevelSampler


def test_uniform_when_empty():
    s = LevelSampler(["A", "B", "C"], history_size=10, exponent=1.0, min_weight=0.0)
    w = s.compute_weights()
    assert pytest.approx(1.0) == sum(w.values())
    assert all(math.isclose(v, 1 / 3) for v in w.values())


def test_inverse_proportional_to_mean_length():
    s = LevelSampler(["A", "B"], history_size=10, exponent=1.0, min_weight=0.0)
    for _ in range(5):
        s.record_episode("A", 100)
    for _ in range(5):
        s.record_episode("B", 400)
    w = s.compute_weights()
    # 1/100 : 1/400 = 4 : 1
    assert pytest.approx(0.8, rel=1e-6) == w["A"]
    assert pytest.approx(0.2, rel=1e-6) == w["B"]


def test_min_weight_floor():
    s = LevelSampler(["A", "B"], history_size=10, exponent=1.0, min_weight=0.1)
    for _ in range(5):
        s.record_episode("A", 1)
    for _ in range(5):
        s.record_episode("B", 10_000)
    w = s.compute_weights()
    assert w["B"] >= 0.1 - 1e-9
    assert pytest.approx(1.0) == sum(w.values())


def test_exponent_sharpens():
    soft = LevelSampler(["A", "B"], exponent=0.5, min_weight=0.0)
    hard = LevelSampler(["A", "B"], exponent=2.0, min_weight=0.0)
    for s in (soft, hard):
        for _ in range(5):
            s.record_episode("A", 100)
        for _ in range(5):
            s.record_episode("B", 400)
    ws = soft.compute_weights()
    wh = hard.compute_weights()
    # Sharper exponent should make the short-level share larger.
    assert wh["A"] > ws["A"]


def test_completed_levels_downweighted():
    s = LevelSampler(["A", "B"], history_size=10, exponent=1.0, min_weight=0.0)
    # Same episode lengths, but A is always completed and B never is.
    for _ in range(5):
        s.record_episode("A", 200, completed=True)
        s.record_episode("B", 200, completed=False)
    w = s.compute_weights()
    assert w["B"] > w["A"]
    # incompletion factors: A=0.05, B=1.05 -> B/A = 21
    assert pytest.approx(21.0, rel=1e-6) == w["B"] / w["A"]
    assert pytest.approx(1.0) == sum(w.values())


def test_completion_rates_reported():
    s = LevelSampler(["A", "B"], history_size=4)
    s.record_episode("A", 100, completed=True)
    s.record_episode("A", 100, completed=False)
    rates = s.completion_rates()
    assert pytest.approx(0.5) == rates["A"]
    assert rates["B"] == 0.0


def test_unobserved_level_uses_fallback_mean():
    s = LevelSampler(["A", "B", "C"], min_weight=0.0)
    for _ in range(5):
        s.record_episode("A", 100)
    for _ in range(5):
        s.record_episode("B", 300)
    # C has no data → treated at the mean of observed (200), not skipped.
    w = s.compute_weights()
    assert w["C"] > 0.0
    assert pytest.approx(1.0) == sum(w.values())
