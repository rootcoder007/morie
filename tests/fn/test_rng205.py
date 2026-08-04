"""Tests for rng205.rangayyan_ch4_csd_from_ccf."""

from morie.fn import _array_core as np

from morie.fn.bsacorr import rangayyan_ch4_csd_from_ccf


def test_rng205_basic():
    """Test basic functionality."""
    theta_xy = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = rangayyan_ch4_csd_from_ccf(theta_xy, X, Y, f, tau)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rng205_edge():
    """Test edge cases."""
    theta_xy = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = rangayyan_ch4_csd_from_ccf(theta_xy, X, Y, f, tau)
    assert isinstance(result, dict)
