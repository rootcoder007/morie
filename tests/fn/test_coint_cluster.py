"""Cointegration + forecasting cluster: _coint core, egcoin engrgr
vecmod, and the four modules the name scan misfiled here (joholt johw
johbu mstrn -- Holt-Winters, hierarchical reconciliation and the
Aalen-Johansen estimator, none of them Johansen cointegration)."""

import numpy as np
import pytest

from morie.fn._coint import adf_test, johansen
from morie.fn.egcoin import engle_granger_2step
from morie.fn.engrgr import engle_granger
from morie.fn.johbu import joseph_bottom_up_reconciliation
from morie.fn.joholt import joseph_holt_linear
from morie.fn.johw import joseph_holt_winters
from morie.fn.mstrn import multistate_transition_matrix
from morie.fn.vecmod import vector_error_correction


def _rw(n, seed):
    return np.cumsum(np.random.default_rng(seed).standard_normal(n))


def test_adf_rejects_stationary_and_not_random_walk():
    rng = np.random.default_rng(0)
    rej_stat = rej_rw = 0
    for seed in range(8):
        r = np.random.default_rng(seed)
        stat = adf_test(r.standard_normal(300))["statistic"]
        rw = adf_test(np.cumsum(r.standard_normal(300)))["statistic"]
        rej_stat += stat < -2.86154  # 5% MacKinnon value, k = 1
        rej_rw += rw < -2.86154
    assert rej_stat == 8  # measured 8/8: white noise is clearly stationary
    assert rej_rw <= 1  # measured 0/8 spurious rejections
    with pytest.raises(ValueError):
        adf_test(np.ones(5))
    with pytest.raises(ValueError):
        adf_test(np.ones(50), trend="quadratic")


def test_engle_granger_finds_a_real_cointegration_and_not_a_fake_one():
    found = spurious = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        x = np.cumsum(rng.standard_normal(300))
        y = 2.0 * x + rng.standard_normal(300)  # cointegrated: I(1) + I(0)
        out = engle_granger_2step(y, x)
        found += out["cointegrated_5pct"]
        assert out["beta"][0] == pytest.approx(2.0, abs=0.15)
        # two independent random walks are NOT cointegrated
        z = np.cumsum(rng.standard_normal(300))
        spurious += engle_granger_2step(z, np.cumsum(rng.standard_normal(300)))[
            "cointegrated_5pct"
        ]
    assert found == 8  # measured 8/8
    assert spurious <= 1  # measured 0/8


def test_engle_granger_uses_mackinnon_not_plain_adf_values():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal(300))
    y = 2 * x + rng.standard_normal(300)
    out = engle_granger_2step(y, x)
    # the k = 2 critical value is stricter than the plain ADF k = 1 one
    assert out["n_vars"] == 2
    assert out["critical_values"][0.05] == pytest.approx(-3.33613)
    assert out["critical_values"][0.05] < -2.86154
    assert out["p_value_band"] in ("< 0.01", "0.01 - 0.05", "0.05 - 0.10", "> 0.10")
    assert engle_granger(y, x)["adf_stat"] == pytest.approx(out["adf_stat"])


def _coint_system(seed=0, n=400):
    """Two series sharing one common trend: rank should be exactly 1."""
    rng = np.random.default_rng(seed)
    trend = np.cumsum(rng.standard_normal(n))
    return np.column_stack([trend + rng.standard_normal(n) * 0.5,
                            2 * trend + rng.standard_normal(n) * 0.5])


def test_johansen_recovers_the_cointegrating_rank():
    ranks = [johansen(_coint_system(seed=s))["rank_5pct"] for s in range(6)]
    assert sum(r == 1 for r in ranks) >= 5  # measured 6/6
    # two independent random walks: rank 0
    zero = 0
    for s in range(6):
        rng = np.random.default_rng(100 + s)
        Y = np.column_stack([np.cumsum(rng.standard_normal(400)),
                             np.cumsum(rng.standard_normal(400))])
        zero += johansen(Y)["rank_5pct"] == 0
    assert zero >= 5  # measured 6/6
    out = johansen(_coint_system())
    assert np.all(np.diff(out["eigenvalues"]) <= 1e-12)  # sorted descending
    assert np.all(out["eigenvalues"] >= 0)
    # trace statistics are decreasing in r by construction
    assert np.all(np.diff(out["trace_stat"]) <= 1e-8)
    with pytest.raises(ValueError):
        johansen(_coint_system()[:, :1])


def test_vecm_alpha_points_back_to_equilibrium():
    Y = _coint_system(seed=2)
    out = vector_error_correction(Y, r=1)
    assert out["alpha"].shape == (2, 1)
    assert out["beta"].shape == (2, 1)
    assert out["ect"].shape[1] == 1
    assert len(out["gamma"]) == 1
    assert out["johansen_rank_5pct"] == 1
    # at least one series must adjust: all-zero alpha would mean no
    # error correction at all despite a cointegrating relation
    assert np.max(np.abs(out["alpha"])) > 0.01
    assert np.all(np.isfinite(out["residuals"]))
    with pytest.raises(ValueError):
        vector_error_correction(Y, r=5)


def test_holt_extrapolates_a_linear_trend():
    y = np.arange(40, dtype=float) * 2.0 + 5.0
    out = joseph_holt_linear(y, horizon=5)
    # a perfectly linear series must be extrapolated at the same slope
    assert out["forecast"] == pytest.approx(
        y[-1] + np.arange(1, 6) * 2.0, rel=0.02
    )
    assert out["sse"] < 1e-3
    # damping must flatten the long horizon relative to the plain trend
    plain = joseph_holt_linear(y, horizon=30)["forecast"][-1]
    damped = joseph_holt_linear(y, horizon=30, damped=True, phi=0.9)["forecast"][-1]
    assert damped < plain
    with pytest.raises(ValueError):
        joseph_holt_linear(y, alpha=1.5)
    with pytest.raises(ValueError):
        joseph_holt_linear(y[:3])


def test_holt_winters_recovers_a_seasonal_pattern():
    m = 12
    season = np.array([3.0, 1.0, -2.0, -4.0, -1.0, 2.0, 5.0, 4.0, 1.0, -1.0, -3.0, -5.0])
    rng = np.random.default_rng(0)
    y = np.concatenate(
        [10 + 0.5 * np.arange(i * m, (i + 1) * m) + season for i in range(8)]
    ) + rng.normal(0, 0.5, 96)
    out = joseph_holt_winters(y, m=m, horizon=12)
    # The forecast must reproduce the seasonal shape. It also carries
    # the trend, so the trend has to come out before comparing --
    # subtracting the mean alone leaves a linear ramp that drags the
    # correlation down to ~0.75 even when the seasonal part is right.
    fc = out["forecast"]
    tt = np.arange(12.0)
    fc_detrended = fc - np.polyval(np.polyfit(tt, fc, 1), tt)
    assert np.corrcoef(fc_detrended, season)[0, 1] > 0.9  # measured 0.954
    assert out["sse"] / y.size < 3.0
    with pytest.raises(ValueError):
        joseph_holt_winters(y[:10], m=12)  # fewer than 2 periods
    # multiplicative refuses non-positive data instead of returning inf
    with pytest.raises(ValueError):
        joseph_holt_winters(np.r_[y, 0.0, -1.0], m=m, seasonal="multiplicative")


def test_reconciliation_is_coherent_and_ols_uses_the_aggregates():
    # 2 bottom series, 1 total
    S = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    bu = joseph_bottom_up_reconciliation([3.0, 4.0], S)
    assert bu["reconciled"] == pytest.approx([7.0, 3.0, 4.0])
    # an incoherent base vector (total != sum of parts) must be repaired
    base = np.array([10.0, 3.0, 4.0])
    ols = joseph_bottom_up_reconciliation(None, S, base=base, method="ols")
    rec = ols["reconciled"]
    assert rec[0] == pytest.approx(rec[1] + rec[2])  # coherent
    assert rec[0] > 7.0  # the aggregate's information was used, unlike bottom-up
    R = np.random.default_rng(0).standard_normal((50, 3)) * np.array([1.0, 5.0, 0.1])
    wls = joseph_bottom_up_reconciliation(None, S, base=base, method="wls", residuals=R)
    assert wls["reconciled"][0] == pytest.approx(
        wls["reconciled"][1] + wls["reconciled"][2]
    )
    with pytest.raises(ValueError):
        joseph_bottom_up_reconciliation(None, S, method="ols")
    with pytest.raises(ValueError):
        joseph_bottom_up_reconciliation([1.0], S)


def test_aalen_johansen_rows_are_probabilities():
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    sf = np.array([0, 0, 0, 1, 0, 1])
    st = np.array([1, 1, 2, 2, 1, 2])
    out = multistate_transition_matrix(time, sf, st, n_states=3)
    P = out["P"]
    assert P.shape == (3, 3)
    assert np.allclose(P.sum(axis=1), 1.0)  # rows are probabilities
    assert np.all(P >= -1e-12)
    # each increment matrix has zero row sums by construction
    for dA in out["increments"]:
        assert np.allclose(dA.sum(axis=1), 0.0)
    assert out["event_times"].size == 6
    with pytest.raises(ValueError):
        multistate_transition_matrix(time, sf, st, n_states=2)  # label out of range
    with pytest.raises(ValueError):
        multistate_transition_matrix(time[:1], sf[:1], st[:1])
