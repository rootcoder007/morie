"""Tests for thinpls.thin_plate_spline."""

from morie.fn import _array_core as np
from morie.fn.thinpls import thin_plate_spline


def test_thinpls_basic():
    """Residuals of the smoothing spline are lam times the kernel weights."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    lam = 0.1
    result = thin_plate_spline(list(zip(x.tolist(), y.tolist())), z, lam=lam)
    assert len(result["fitted"]) == 100
    assert max(abs(r - lam * w) for r, w in zip(result["residuals"], result["w"])) < 1e-9
    assert 3.0 < result["df"] < 100.0


def test_thinpls_edge():
    """lam = 0 interpolates the data exactly."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = thin_plate_spline(list(zip(x.tolist(), y.tolist())), z, lam=0.0)
    assert max(abs(r) for r in result["residuals"]) < 1e-8
    assert result["df"] == 100.0
