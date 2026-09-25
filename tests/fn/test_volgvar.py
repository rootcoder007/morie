"""Tests for volgvar.vol_garch_var_backtest."""

import math

import pytest

from morie.fn import _stats_core as stats
from morie.fn.volcc import vol_christoffersen_cc
from morie.fn.volgvar import vol_garch_var_backtest

HITS = [0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0]


def test_volgvar_reports_the_whole_triple():
    """statistic/pvalue carry LR_cc, and the two components come through
    unchanged so a caller can see WHICH half of the null fails."""
    out = vol_garch_var_backtest(HITS, 0.05)
    cc = vol_christoffersen_cc(HITS, alpha=0.05)
    assert out["statistic"] == pytest.approx(float(cc["statistic"]), rel=1e-12)
    assert out["pvalue"] == pytest.approx(float(cc["pvalue"]), rel=1e-12)
    assert out["lr_cc"] == pytest.approx(float(cc["statistic"]), rel=1e-12)
    assert out["pvalue_cc"] == pytest.approx(float(cc["pvalue"]), rel=1e-12)
    assert out["lr_uc"] == pytest.approx(float(cc["lr_uc"]), rel=1e-12)
    assert out["lr_ind"] == pytest.approx(float(cc["lr_ind"]), rel=1e-12)
    # LR_cc is the sum of the two halves it reports
    assert out["lr_cc"] == pytest.approx(out["lr_uc"] + out["lr_ind"], rel=1e-12)
    assert out["pvalue_uc"] == pytest.approx(
        float(stats.chi2.sf(out["lr_uc"], 1)), rel=1e-12)
    assert out["pvalue_ind"] == pytest.approx(
        float(stats.chi2.sf(out["lr_ind"], 1)), rel=1e-12)
    assert out["n_obs"] == 12 and out["n_exceedances"] == 5


def test_volgvar_separates_a_calibration_failure_from_a_dynamics_failure():
    """Right rate, clustered breaches: the independence half rejects while
    the coverage half is exactly zero. The reverse for a rate that is
    four times the claim but perfectly spread."""
    clustered = vol_garch_var_backtest([1] * 10 + [0] * 90, 0.10)
    assert clustered["lr_uc"] == pytest.approx(0.0, abs=1e-12)
    assert clustered["pvalue_uc"] == pytest.approx(1.0, rel=1e-12)
    assert clustered["pvalue_ind"] < 0.01
    assert clustered["pvalue_cc"] < 0.05

    spread = vol_garch_var_backtest(([1] + [0] * 4) * 20, 0.05)
    assert spread["n_exceedances"] == 20
    assert spread["pvalue_uc"] < 1e-6
    # every breach is isolated, so the chain says nothing about clustering
    assert spread["lr_ind"] < spread["lr_uc"]


def test_volgvar_zero_breaches_is_pure_coverage():
    out = vol_garch_var_backtest([0] * 40, 0.05)
    assert out["lr_ind"] == pytest.approx(0.0, abs=1e-12)
    assert out["lr_uc"] == pytest.approx(-2.0 * 40 * math.log(0.95), rel=1e-12)
    assert out["statistic"] == pytest.approx(out["lr_uc"], rel=1e-12)


def test_volgvar_rejects_bad_input():
    with pytest.raises(ValueError, match="at least 2"):
        vol_garch_var_backtest([1])
    with pytest.raises(ValueError, match="0/1"):
        vol_garch_var_backtest([0, 1, 3, 0])
