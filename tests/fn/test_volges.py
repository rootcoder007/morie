"""Tests for volges."""

import pytest
from morie.fn import _stats_core as stats

from morie.fn.volges import vol_garch_es_impl


def test_volges_basic():
    out = vol_garch_es_impl(0.0, 1.0, alpha=0.05)
    z = stats.norm.ppf(0.05)
    assert out["es"] == pytest.approx(stats.norm.pdf(z) / 0.05, abs=1e-12)
    assert out["es"] > out["var"]  # ES always lies beyond VaR


def test_volges_edge():
    with pytest.raises(ValueError):
        vol_garch_es_impl(0.0, 1.0, dist="t", nu=1.5)  # infinite variance
    with pytest.raises(ValueError):
        vol_garch_es_impl(0.0, 1.0, dist="cauchy")
