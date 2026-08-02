"""Tests for gb_psi (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb_psi import gibbons_pitman_efficiency


def test_gb_psi_basic():
    from scipy import stats
    out = gibbons_pitman_efficiency(
        lambda x: stats.ttest_1samp(x, 0.0).pvalue,
        lambda x: stats.ttest_1samp(x, 0.0).pvalue,
        lambda th, n, rng: rng.standard_normal(n) + th,
        delta=0.4, n=50, n_sim=100)
    assert out["efficiency_ratio"] == pytest.approx(1.0)  # same test -> ratio 1


def test_gb_psi_edge():
    with pytest.raises(ValueError):
        gibbons_pitman_efficiency(None, None, None, n=2)
