"""Tests for gb433 (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb433 import gibbons_ks_kolmogorov_limit


def test_gb433_basic():
    from scipy import stats
    assert gibbons_ks_kolmogorov_limit(1.0)["L"] == pytest.approx(
        stats.kstwobign.cdf(1.0), abs=1e-10)


def test_gb433_edge():
    with pytest.raises(ValueError):
        gibbons_ks_kolmogorov_limit(0.0)
