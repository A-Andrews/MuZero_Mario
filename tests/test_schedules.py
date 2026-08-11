"""Piecewise-linear schedules shared by the imitation mix and the eps anneal."""
import pytest

from src.muzero.schedules import validate_schedule, value_at


def test_empty_schedule_is_none():
    assert validate_schedule(None) is None
    assert validate_schedule([]) is None


def test_validate_coerces_types():
    pts = validate_schedule([[0, 1], [10, 0]])
    assert pts == [(0, 1.0), (10, 0.0)]
    assert all(isinstance(s, int) and isinstance(v, float) for s, v in pts)


def test_validate_rejects_non_ascending_steps():
    with pytest.raises(AssertionError):
        validate_schedule([[10, 0.2], [10, 0.1]])
    with pytest.raises(AssertionError):
        validate_schedule([[10, 0.2], [5, 0.1]])


def test_validate_enforces_bounds_only_when_given():
    with pytest.raises(AssertionError):
        validate_schedule([[0, 1.5]], lo=0.0, hi=1.0)
    assert validate_schedule([[0, 1.5]]) == [(0, 1.5)]


def test_value_clamps_outside_the_knot_range():
    pts = validate_schedule([[100, 0.25], [300, 0.05]])
    assert value_at(0, pts) == 0.25
    assert value_at(100, pts) == 0.25
    assert value_at(300, pts) == 0.05
    assert value_at(10**9, pts) == 0.05


def test_value_interpolates_linearly():
    pts = validate_schedule([[0, 0.25], [400, 0.05]])
    assert value_at(200, pts) == pytest.approx(0.15)
    assert value_at(100, pts) == pytest.approx(0.20)


def test_knots_return_stored_values_exactly():
    """No float residue at the knots — the anneal should hit 0.05 exactly."""
    pts = validate_schedule([[0, 0.25], [7, 0.05], [11, 0.0]])
    assert value_at(7, pts) == 0.05
    assert value_at(11, pts) == 0.0


def test_single_point_schedule_is_constant():
    pts = validate_schedule([[500, 0.1]])
    assert value_at(0, pts) == 0.1
    assert value_at(10**6, pts) == 0.1


def test_matches_the_shipped_specialist_anneal():
    """The grid submit_specialist.sh actually passes."""
    pts = validate_schedule([[0, 0.25], [500_000, 0.05]], lo=0.0, hi=1.0)
    assert value_at(0, pts) == 0.25
    assert value_at(250_000, pts) == pytest.approx(0.15)
    assert value_at(500_000, pts) == 0.05
    assert value_at(780_000, pts) == 0.05  # past the diag runs' final step
