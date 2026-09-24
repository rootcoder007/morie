"""Tests for sunabr.sun_abraham_did."""

from morie.fn import _array_core as np

from morie.fn.sunabr import sun_abraham_did


def test_sunabr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    unit = np.random.default_rng(42).normal(0.0, 1.0, 40)
    time = np.random.default_rng(42).normal(0.0, 1.0, 40)
    cohort = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = sun_abraham_did(y, unit, time, cohort)
    assert isinstance(result, dict)
    assert "event_time" in result


def test_sunabr_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    unit = np.random.default_rng(42).normal(0.0, 1.0, 40)
    time = np.random.default_rng(42).normal(0.0, 1.0, 40)
    cohort = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = sun_abraham_did(y, unit, time, cohort)
    assert isinstance(result, dict)
