"""Tests for rgexecg.rangayyan_exercise_ecg."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_exercise_ecg


def test_rgexecg_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    qrs = 5
    fs = 0.1
    result = rangayyan_exercise_ecg(x, qrs, fs)
    assert isinstance(result, dict)
    assert "stdev" in result


def test_rgexecg_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    qrs = 5
    fs = 0.1
    result = rangayyan_exercise_ecg(x, qrs, fs)
    assert isinstance(result, dict)
