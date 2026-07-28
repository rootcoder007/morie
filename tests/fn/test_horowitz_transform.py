"""Transformation models T(Y) = X'beta + U (Horowitz Ch. 6)."""

import numpy as np
import pytest

from morie.fn.hrzchet import horowitz_chen_estimator_T
from morie.fn.hrzhot import horowitz_T_F_estimators
from morie.fn.hrzlam import horowitz_baseline_hazard_est
from morie.fn.hrztfap import horowitz_T_F_asymp_props
from morie.fn.hrzycp import horowitz_conditional_prediction


def _sample(n=400, seed=0):
    """T(y) = log y exactly: Y = exp(X'beta + U)."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.standard_normal(n), rng.standard_normal(n)])
    beta = np.array([1.0, -0.5])
    Y = np.exp(X @ beta + rng.logistic(size=n) * 0.6)
    return X, Y, beta


def test_horowitz_T_recovers_a_known_log_transformation():
    X, Y, beta = _sample()
    out = horowitz_T_F_estimators(X, Y, (0.5, 0.5), beta)
    # T(y0) = 0 by the location normalisation, so the target is
    # log(y) - log(y0), not log(y)
    truth = np.log(out["y_grid"]) - np.log(out["y0"])
    assert np.corrcoef(out["T_hat"], truth)[0, 1] > 0.97
    slope = np.polyfit(truth, out["T_hat"], 1)[0]
    assert abs(slope - 1.0) < 0.25
    assert out["F_is_empirical_cdf"] is False
    assert out["beta"][0] == 1.0  # |b_1| = 1 imposed


def test_horowitz_F_is_a_distribution_function():
    X, Y, beta = _sample()
    out = horowitz_T_F_estimators(X, Y, (0.5, 0.5), beta)
    F = out["F_hat"][~np.isnan(out["F_hat"])]
    assert np.all(F >= 0) and np.all(F <= 1)
    assert np.all(np.diff(F) >= -1e-12)  # non-decreasing


def test_horowitz_T_F_validates_and_normalises():
    X, Y, beta = _sample(60)
    with pytest.raises(ValueError):
        horowitz_T_F_estimators(X, Y, -1.0, beta)
    with pytest.raises(ValueError):
        horowitz_T_F_estimators(X, Y, 0.5, [1.0])
    with pytest.raises(ValueError):
        horowitz_T_F_estimators(X[:10], Y[:10], 0.5, beta)
    # |beta_1| = 1 cannot be imposed when beta_1 = 0
    with pytest.raises(ValueError):
        horowitz_T_F_estimators(X, Y, 0.5, [0.0, 1.0])
    # a doubled beta gives the same normalised beta
    a = horowitz_T_F_estimators(X, Y, 0.6, beta)["beta"]
    b = horowitz_T_F_estimators(X, Y, 0.6, 2 * beta)["beta"]
    assert np.allclose(a, b)


def test_asymptotics_report_a_process_and_the_HT9_bandwidth_split():
    X, Y, _ = _sample(200)
    n = 200
    good = horowitz_T_F_asymp_props(X, Y, (n ** (-1 / 3), n ** (-1 / 10)))
    assert good["limit_is_process"] is True
    assert good["rate_exponent"] == -0.5
    assert good["Kz_order_required"] == 6
    assert good["bandwidths_consistent_with_HT9"] is True
    # h_nz must shrink much more slowly than h_ny; the reference
    # rates differ by a factor of n^{7/30}
    assert good["h_nz_reference"] > good["h_ny_reference"]
    assert good["h_nz_reference"] / good["h_ny_reference"] == pytest.approx(
        n ** (1 / 3 - 1 / 10))
    # using one bandwidth for both violates HT9
    assert horowitz_T_F_asymp_props(
        X, Y, n ** (-1 / 3))["bandwidths_consistent_with_HT9"] is False
    with pytest.raises(ValueError):
        horowitz_T_F_asymp_props(X, Y, 0.0)


def test_chen_estimator_tracks_the_transformation_and_uses_no_kernel():
    X, Y, beta = _sample(250, seed=2)
    out = horowitz_chen_estimator_T(X, Y, beta_hat=beta)
    truth = np.log(out["y_grid"]) - np.log(out["y0"])
    assert np.corrcoef(out["T_hat"], truth)[0, 1] > 0.9
    assert out["uses_kernel"] is False
    # the book compares the two and finds neither dominates
    assert out["faster_than_horowitz"] is False
    assert out["rate_exponent"] == -0.5


def test_chen_and_horowitz_share_the_same_rate():
    X, Y, beta = _sample(200, seed=3)
    a = horowitz_chen_estimator_T(X, Y, beta_hat=beta)
    b = horowitz_T_F_asymp_props(X, Y, (0.3, 0.6))
    assert a["rate_exponent"] == b["rate_exponent"] == -0.5


def test_baseline_hazard_smooths_the_step_function():
    rng = np.random.default_rng(1)
    n = 800
    X = np.column_stack([rng.standard_normal(n), rng.standard_normal(n)])
    beta = np.array([0.5, -0.3])
    # constant baseline hazard 1: exponential durations scaled by the
    # risk score
    t = rng.exponential(1.0, n) / np.exp(X @ beta)
    ev = np.ones(n)
    out = horowitz_baseline_hazard_est(t, X, ev, beta)
    mid = (out["grid"] > np.quantile(out["grid"], 0.2)) & \
          (out["grid"] < np.quantile(out["grid"], 0.6))
    assert abs(np.median(out["lambda0_hat"][mid]) - 1.0) < 0.4
    # the cumulative hazard is a step function: it never decreases
    assert np.all(np.diff(out["cumhaz"]) > 0)
    assert out["root_n_attainable"] is False
    assert out["rate_exponent"] == pytest.approx(-0.4)
    assert out["n_events"] == n


def test_baseline_hazard_handles_censoring_and_validates():
    rng = np.random.default_rng(6)
    n = 300
    X = np.column_stack([rng.standard_normal(n), rng.standard_normal(n)])
    beta = np.array([0.4, 0.0])
    t = rng.exponential(1.0, n)
    ev = (rng.random(n) > 0.3).astype(float)
    out = horowitz_baseline_hazard_est(t, X, ev, beta)
    assert out["n_events"] == int(ev.sum()) < n
    assert np.all(out["lambda0_hat"] >= 0)
    with pytest.raises(ValueError):
        horowitz_baseline_hazard_est(t, X, np.zeros(n), beta)  # no events
    with pytest.raises(ValueError):
        horowitz_baseline_hazard_est(-t, X, ev, beta)
    with pytest.raises(ValueError):
        horowitz_baseline_hazard_est(t, X, ev * 2, beta)


def test_conditional_prediction_gives_a_probability_and_a_quantile():
    yg = np.linspace(0.5, 8.0, 60)
    Tg = np.log(yg)
    ug = np.linspace(-4, 4, 81)
    Fg = 1.0 / (1.0 + np.exp(-ug))     # logistic U
    beta = np.array([1.0, -0.5])
    out = horowitz_conditional_prediction(np.array([0.3, 0.2]), 2.0,
                                          Tg, Fg, beta,
                                          y_grid=yg, u_grid=ug)
    z = 0.3 * 1.0 + 0.2 * -0.5
    assert out["probability"] == pytest.approx(
        1 / (1 + np.exp(-(np.log(2.0) - z))), abs=1e-3)
    # the median predictor solves T(y) = x'b, i.e. y = exp(x'b)
    assert out["gamma"] == 0.5
    assert abs(out["u_gamma"]) < 0.15
    assert abs(out["quantile"] - np.exp(z)) < 0.3
    # the section's real result: the MEAN is not root-n estimable
    assert out["mean_root_n_estimable"] is False
    assert out["quantile_root_n_estimable"] is True


def test_conditional_prediction_validates_its_inputs():
    yg = np.linspace(0.5, 8.0, 40)
    ug = np.linspace(-4, 4, 41)
    beta = np.array([1.0, -0.5])
    with pytest.raises(ValueError):
        horowitz_conditional_prediction([0.3, 0.2], 2.0, np.log(yg),
                                        np.linspace(0, 1, 41), beta,
                                        y_grid=yg, u_grid=ug, gamma=0.0)
    with pytest.raises(ValueError):
        horowitz_conditional_prediction([0.3, 0.2], 2.0, np.log(yg),
                                        np.linspace(0, 2, 41), beta,
                                        y_grid=yg, u_grid=ug)
    with pytest.raises(ValueError):  # T must be increasing
        horowitz_conditional_prediction([0.3, 0.2], 2.0, -np.log(yg),
                                        np.linspace(0, 1, 41), beta,
                                        y_grid=yg, u_grid=ug)
    with pytest.raises(ValueError):  # y_grid required for an array T
        horowitz_conditional_prediction([0.3, 0.2], 2.0, np.log(yg),
                                        np.linspace(0, 1, 41), beta,
                                        u_grid=ug)
