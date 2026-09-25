"""Tests for volkupiec.vol_kupiec_var_test."""

import math

import pytest

from morie.fn import _stats_core as stats
from morie.fn.volkupiec import vol_kupiec_var_test


def _lr_uc(p, t, n):
    """The docstring's LR_uc, written out in plain arithmetic."""
    phat = n / t
    ll_null = (t - n) * math.log(1.0 - p) + (n * math.log(p) if n else 0.0)
    ll_alt = ((t - n) * math.log(1.0 - phat) if n < t else 0.0) + (
        n * math.log(phat) if n else 0.0
    )
    return -2.0 * (ll_null - ll_alt)


def test_volkupiec_vanishes_when_the_rate_equals_the_claim():
    """N/T == alpha puts the null AT the unrestricted MLE, so LR_uc is
    exactly zero and the p-value exactly one."""
    hits = [0] * 95 + [1] * 5
    out = vol_kupiec_var_test(hits, 0.05)
    assert out["statistic"] == pytest.approx(0.0, abs=1e-12)
    assert out["pvalue"] == pytest.approx(1.0, rel=1e-12)
    assert out["n_obs"] == 100
    assert out["n_exceedances"] == 5
    assert out["expected_exceedances"] == pytest.approx(5.0, rel=1e-12)
    assert out["rate"] == pytest.approx(0.05, rel=1e-12)
    assert out["df"] == 1


def test_volkupiec_matches_the_likelihood_ratio_arithmetic():
    """Four times the claimed rate: recompute LR_uc and its chi2_1 tail."""
    hits = [1] * 20 + [0] * 80
    out = vol_kupiec_var_test(hits, 0.05)
    want = _lr_uc(0.05, 100, 20)
    assert out["statistic"] == pytest.approx(want, rel=1e-12)
    assert out["pvalue"] == pytest.approx(float(stats.chi2.sf(want, 1)), rel=1e-12)
    assert out["rate"] == pytest.approx(0.20, rel=1e-12)
    # 20 breaches where 5 were promised is a decisive rejection
    assert out["pvalue"] < 1e-6


def test_volkupiec_zero_exceedances_uses_the_log_form():
    """N = 0 makes the alternative likelihood 1, so LR_uc = -2 T log(1-p)
    rather than log(0); the same for N = T."""
    none = vol_kupiec_var_test([0] * 50, 0.05)
    assert none["statistic"] == pytest.approx(-2.0 * 50 * math.log(0.95), rel=1e-12)
    assert none["n_exceedances"] == 0
    assert none["rate"] == 0.0
    every = vol_kupiec_var_test([1] * 50, 0.05)
    assert every["statistic"] == pytest.approx(-2.0 * 50 * math.log(0.05), rel=1e-12)
    assert every["rate"] == 1.0


def test_volkupiec_is_coverage_only_and_ignores_order():
    """Clustering every breach together leaves LR_uc untouched -- the
    docstring's point that this test sees the rate and nothing else."""
    spread = vol_kupiec_var_test(([1] + [0] * 9) * 10, 0.05)
    clustered = vol_kupiec_var_test([1] * 10 + [0] * 90, 0.05)
    assert spread["statistic"] == pytest.approx(clustered["statistic"], rel=1e-12)
    assert spread["statistic"] == pytest.approx(_lr_uc(0.05, 100, 10), rel=1e-12)


def test_volkupiec_rejects_bad_input():
    with pytest.raises(ValueError, match="at least 2"):
        vol_kupiec_var_test([1])
    with pytest.raises(ValueError, match="0/1"):
        vol_kupiec_var_test([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="strictly between"):
        vol_kupiec_var_test([0, 1, 0, 0], alpha=0.0)
