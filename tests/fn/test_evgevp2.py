"""Tests for evgevp2.evt_gev_pwm."""

from morie.fn import _array_core as np
from morie.fn.evgevp2 import evt_gev_pwm


def test_evgevp2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gev_pwm(x)
    assert isinstance(result, dict)
    assert "mu" in result


def test_evgevp2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gev_pwm(x)
    assert isinstance(result, dict)


def test_pwm_fit_is_the_lmoment_fit_with_the_pwms_reported():
    import pytest

    from morie.fn.evgevlm import ev_gev_lmoments

    x = [21.0, 34.5, 19.2, 55.1, 28.3, 31.0, 44.2, 25.5, 38.8, 27.1, 60.4, 23.9]
    xs = sorted(x)
    n = 12
    b1 = sum((j - 1) / (n - 1) * v for j, v in enumerate(xs, start=1)) / n
    r = evt_gev_pwm(x)
    ref = ev_gev_lmoments(x)
    assert r["mu"] == ref["mu"] and r["k_hosking"] == ref["k_hosking"]
    assert r["b0"] == pytest.approx(sum(x) / n, rel=1e-13)
    assert r["b1"] == pytest.approx(b1, rel=1e-13)
    assert r["l2"] == pytest.approx(2 * r["b1"] - r["b0"], rel=1e-12)
