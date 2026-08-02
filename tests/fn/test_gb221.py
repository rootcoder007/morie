"""Tests for gb221 (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb221 import gibbons_quantile_deriv


def test_gb221_basic():
    from scipy import stats
    out = gibbons_quantile_deriv(0.8, stats.norm())
    assert out["Q_prime"] == pytest.approx(1 / stats.norm.pdf(stats.norm.ppf(0.8)), rel=1e-8)


def test_gb221_edge():
    with pytest.raises(ValueError):
        gibbons_quantile_deriv(1.5, None)
