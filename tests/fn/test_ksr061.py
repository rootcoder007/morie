"""Tests for ksr061 (Kosorok shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ksr061 import kosorok_ch3_differentiable_quadratic_mean


def test_ksr061_basic():
    from scipy import stats
    out = kosorok_ch3_differentiable_quadratic_mean(
        lambda x, th: float(stats.norm.pdf(x, loc=th)), lambda x: float(x))
    assert out["shrinking"] is True


def test_ksr061_edge():
    from scipy import stats
    with pytest.raises(ValueError):
        kosorok_ch3_differentiable_quadratic_mean(
            lambda x, th: float(stats.norm.pdf(x, loc=th)), lambda x: float(x),
            t_grid=[0.0])
