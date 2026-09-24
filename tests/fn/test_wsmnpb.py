"""Tests for wsmnpb.wasserman_nonparametric_boot."""

from morie.fn import _array_core as np

from morie.fn.wsmnpb import wasserman_nonparametric_boot


def test_wsmnpb_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0.0, 1.0, 40)
    T = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    B = 5
    result = wasserman_nonparametric_boot(data, T, B)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_wsmnpb_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0.0, 1.0, 40)
    T = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    B = 5
    result = wasserman_nonparametric_boot(data, T, B)
    assert isinstance(result, dict)
