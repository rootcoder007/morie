"""Tests for rng017.rangayyan_ch3_acf_ensemble_estimate."""

from morie.fn import _array_core as np

from morie.fn.bsacorr import rangayyan_ch3_acf_ensemble_estimate


def test_rng017_basic():
    """Test basic functionality."""
    x_k = np.random.default_rng(42).normal(0.0, 1.0, 40)
    t1 = 5
    tau = 5
    result = rangayyan_ch3_acf_ensemble_estimate(x_k, t1, tau)
    assert isinstance(result, dict)
    assert "acf" in result


def test_rng017_edge():
    """Test edge cases."""
    x_k = np.random.default_rng(42).normal(0.0, 1.0, 40)
    t1 = 5
    tau = 5
    result = rangayyan_ch3_acf_ensemble_estimate(x_k, t1, tau)
    assert isinstance(result, dict)
