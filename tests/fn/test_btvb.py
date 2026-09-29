"""Tests for btvb.boot_var_estimator."""

from morie.fn import _array_core as np
from morie.fn.btvb import boot_var_estimator


def test_btvb_basic():
    """Test basic functionality."""
    theta_b = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_var_estimator(theta_b)
    assert isinstance(result, dict)
    assert "value" in result


def test_btvb_edge():
    """Test edge cases."""
    theta_b = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_var_estimator(theta_b)
    assert isinstance(result, dict)


def test_btvb_is_the_b_minus_1_variance():
    import math

    import pytest

    reps = [1.2, 0.8, 1.5, 1.1, 0.9, 1.3]
    m = sum(reps) / 6
    v = sum((t - m) ** 2 for t in reps) / 5
    r = boot_var_estimator(reps)
    assert r["value"] == pytest.approx(v, rel=1e-13)
    assert r["se"] == pytest.approx(math.sqrt(v), rel=1e-13)
    assert r["mean_replicate"] == pytest.approx(m, rel=1e-14)
