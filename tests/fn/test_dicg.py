"""Tests for dicg.deviance_information_criterion."""

from morie.fn import _array_core as np
from morie.fn.dicg import deviance_information_criterion


def test_dicg_basic():
    """Test basic functionality."""
    deviance = np.random.default_rng(42).normal(0, 1, 100)
    result = deviance_information_criterion(deviance)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_dicg_edge():
    """Test edge cases."""
    deviance = np.random.default_rng(42).normal(0, 1, 100)
    result = deviance_information_criterion(deviance)
    assert isinstance(result, dict)


def test_dic_both_complexity_variants_recomputed():
    import pytest

    d = [102.0, 98.5, 101.2, 99.8, 100.4, 97.9]
    dbar = sum(d) / 6
    pv = 0.5 * sum((v - dbar) ** 2 for v in d) / 5
    r = deviance_information_criterion(d)
    assert r["estimate"] == pytest.approx(dbar + pv, rel=1e-14)
    r2 = deviance_information_criterion(d, d_at_mean=96.0)
    assert r2["p_d"] == pytest.approx(dbar - 96.0, rel=1e-14)
    assert r2["estimate"] == pytest.approx(96.0 + 2 * (dbar - 96.0), rel=1e-14)
