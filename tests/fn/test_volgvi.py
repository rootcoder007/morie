"""Tests for volgvi."""

import pytest
from scipy import stats

from morie.fn.volgvi import vol_garch_var_impl


def test_volgvi_basic():
    out = vol_garch_var_impl(0.0, 1.0, alpha=0.05)
    assert out["var"] == pytest.approx(-stats.norm.ppf(0.05), abs=1e-12)
    assert vol_garch_var_impl(0.0, 3.0)["var"] == pytest.approx(3 * out["var"])


def test_volgvi_edge():
    # a fatter tail must widen the 1% VaR
    assert (
        vol_garch_var_impl(0.0, 1.0, 0.01, dist="t", nu=4.0)["var"]
        > vol_garch_var_impl(0.0, 1.0, 0.01)["var"]
    )
    with pytest.raises(ValueError):
        vol_garch_var_impl(0.0, 0.0)
    with pytest.raises(ValueError):
        vol_garch_var_impl(0.0, 1.0, alpha=0.0)
