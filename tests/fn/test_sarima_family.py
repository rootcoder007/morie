"""Seasonal ARIMA and regression with seasonal ARIMA errors."""
import importlib
import math

import pytest

S = importlib.import_module("morie.fn.sarima")
SX = importlib.import_module("morie.fn.sarimax")

Z = S.series_g(log=True)
W = S.difference(Z, 1, 1, 12)


@pytest.fixture(scope="module")
def ml():
    return S.fit(Z, (0, 1, 1), (0, 1, 1), 12, method="ml")


def test_series_g_is_the_books_table():
    g = S.series_g()
    assert len(g) == 144
    assert g[:3] == [112.0, 118.0, 132.0]
    assert g[-1] == 432.0


def test_logging_the_series_is_offered_not_assumed():
    assert S.series_g(log=True)[0] == pytest.approx(math.log(112.0))


def test_differencing_counts():
    assert len(S.difference(Z, 1, 0, 12)) == 143
    assert len(S.difference(Z, 0, 1, 12)) == 132
    assert len(W) == 131


def test_seasonal_differencing_needs_a_period():
    with pytest.raises(ValueError):
        S.difference(Z, 0, 1, 1)


def test_a_short_series_is_refused():
    with pytest.raises(ValueError):
        S.difference([1.0, 2.0, 3.0], 1, 1, 12)


def test_the_sample_acf_matches_the_printed_values():
    r = S.sample_acf(W, (1, 12))
    assert r[1] == pytest.approx(-0.34, abs=0.005)
    assert r[12] == pytest.approx(-0.39, abs=0.005)


def test_a_lag_out_of_range_is_refused():
    with pytest.raises(ValueError):
        S.sample_acf(W, (0,))
    with pytest.raises(ValueError):
        S.sample_acf(W, (len(W),))


def test_a_constant_series_has_no_autocorrelation_to_report():
    with pytest.raises(ValueError):
        S.sample_acf([1.0] * 20, (1,))


def test_airline_autocovariances_vanish_off_the_four_lags():
    ac = S.airline_autocovariances(0.4, 0.6)
    assert set(ac["gamma"]) == {0, 1, 11, 12, 13}
    assert ac["gamma"][11] == pytest.approx(0.24)
    assert ac["gamma"][11] == ac["gamma"][13]


def test_gamma_zero_factorises():
    ac = S.airline_autocovariances(0.4, 0.6, sigma2=2.0)
    assert ac["gamma"][0] == pytest.approx(
        (1 + 0.16) * (1 + 0.36) * 2.0)


def test_rho_one_is_free_of_the_seasonal_parameter():
    a = S.airline_autocovariances(0.4, 0.1)
    b = S.airline_autocovariances(0.4, 0.9)
    assert a["rho_1"] == pytest.approx(b["rho_1"])
    assert a["rho_12"] != pytest.approx(b["rho_12"])


def test_moment_estimate_solves_the_printed_values():
    assert S.moment_estimate(-0.34) == pytest.approx(0.39, abs=0.005)
    assert S.moment_estimate(-0.39) == pytest.approx(0.48, abs=0.005)


def test_moment_estimate_refuses_an_impossible_correlation():
    with pytest.raises(ValueError):
        S.moment_estimate(-0.6)


def test_expanding_the_operators_gives_equation_9_2_2():
    ar, ma = S.expand_polynomials((), (), (0.4,), (0.6,), 12)
    assert ar == []
    assert ma[0] == pytest.approx(0.4)
    assert ma[11] == pytest.approx(0.6)
    assert ma[12] == pytest.approx(-0.24)
    assert all(abs(v) < 1e-15 for v in ma[1:11])


def test_seasonal_terms_need_a_period():
    with pytest.raises(ValueError):
        S.expand_polynomials((), (0.5,), (), (), 1)


def test_exact_ml_reproduces_the_r_output(ml):
    r = S.r_convention(ml)
    assert r["ma"][0] == pytest.approx(-0.4018, abs=5e-5)
    assert r["sma"][0] == pytest.approx(-0.5569, abs=5e-5)
    assert ml["sigma2"] == pytest.approx(0.001348, abs=5e-7)
    assert ml["loglik"] == pytest.approx(244.7, abs=0.05)
    assert ml["aic"] == pytest.approx(-483.4, abs=0.1)


def test_the_exact_sum_of_squares_gives_the_books_ls_estimates():
    u = S.fit(Z, (0, 1, 1), (0, 1, 1), 12, method="uls")
    assert u["theta"][0] == pytest.approx(0.40, abs=0.02)
    assert u["Theta"][0] == pytest.approx(0.61, abs=0.02)
    assert u["sigma2"] == pytest.approx(1.34e-3, abs=2e-5)


def test_the_likelihood_ranks_the_routes(ml):
    u = S.fit(Z, (0, 1, 1), (0, 1, 1), 12, method="uls")
    c = S.fit(Z, (0, 1, 1), (0, 1, 1), 12, method="css")
    assert ml["loglik"] > u["loglik"]
    assert ml["loglik"] > c["loglik"]


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError):
        S.fit(Z, (0, 1, 1), (0, 1, 1), 12, method="wishful")


def test_the_moment_route_is_defined_for_the_airline_model_only():
    with pytest.raises(ValueError):
        S.fit(Z, (1, 1, 1), (0, 1, 1), 12, method="moment")


def test_a_model_with_no_parameters_is_refused():
    with pytest.raises(ValueError):
        S.fit(Z, (0, 1, 0), (0, 1, 0), 12)


def test_the_initial_covariance_solves_the_lyapunov_equation(ml):
    ar, ma = S.expand_polynomials((), (), ml["theta"], ml["Theta"], 12)
    T, R, r = S._state_space(ar, ma)
    P = S._initial_covariance(T, R, r)
    TP = [[sum(T[i][k] * P[k][j] for k in range(r)) for j in range(r)]
          for i in range(r)]
    rhs = [[sum(TP[i][k] * T[j][k] for k in range(r)) + R[i] * R[j]
            for j in range(r)] for i in range(r)]
    for i in range(r):
        for j in range(r):
            assert P[i][j] == pytest.approx(rhs[i][j], abs=1e-10)


def test_the_prediction_variance_starts_at_the_ma_variance(ml):
    ar, ma = S.expand_polynomials((), (), ml["theta"], ml["Theta"], 12)
    T, R, r = S._state_space(ar, ma)
    P = S._initial_covariance(T, R, r)
    th, TH = ml["theta"][0], ml["Theta"][0]
    assert P[0][0] == pytest.approx((1 + th * th) * (1 + TH * TH))


def test_css_residuals_reproduce_the_recursion():
    ar, ma = S.expand_polynomials((), (), (0.4,), (0.6,), 12)
    r = S.css(W, ar, ma, full=True)
    a = r["residuals"]
    for t in (0, 1, 20, 130):
        pred = 0.0
        for j, c in enumerate(ma):
            if t - j - 1 >= 0:
                pred -= c * a[t - j - 1]
        assert a[t] == pytest.approx(W[t] - pred)


def test_forecasts_repeat_the_seasonal_shape(ml):
    f = S.forecast(ml, 36)
    assert len(f["forecast"]) == 36
    assert all(f["forecast"][k + 12] > f["forecast"][k]
               for k in range(24))


def test_forecast_intervals_widen(ml):
    f = S.forecast(ml, 24)
    assert f["se"][0] == pytest.approx(math.sqrt(ml["sigma2"]))
    assert all(f["se"][i] < f["se"][i + 1] for i in range(23))


def test_a_non_positive_horizon_is_refused(ml):
    with pytest.raises(ValueError):
        S.forecast(ml, 0)


def test_large_sample_variances_match_the_book():
    se = S.large_sample_se(0.40, 0.61, 131)
    assert se["var_theta"] == pytest.approx(0.0064, abs=5e-5)
    assert se["var_Theta"] == pytest.approx(0.0048, abs=5e-5)
    assert se["cov"] == 0.0


def test_bartlett_gives_the_books_standard_error():
    r = S.sample_acf(W, (1, 12))
    b = S.bartlett_se({1: r[1], 11: 0.0, 12: r[12], 13: 0.0}, 131)
    assert b["se"] == pytest.approx(0.11, abs=0.005)
    assert b["se"] > b["white_noise_se"]


def test_bartlett_refuses_a_non_positive_sample_size():
    with pytest.raises(ValueError):
        S.bartlett_se({1: 0.0, 11: 0.0, 12: 0.0, 13: 0.0}, 0)


# ------------------------------------------------------------ sarimax
def test_no_regressors_reduces_to_sarima(ml):
    r = SX.fit(Z, None, (0, 1, 1), (0, 1, 1), 12)
    assert r["theta"][0] == pytest.approx(ml["theta"][0], abs=1e-4)
    assert r["loglik"] == pytest.approx(ml["loglik"], abs=1e-6)


def test_a_planted_coefficient_is_recovered_within_two_standard_errors():
    xs = [math.sin(i * 0.7) for i in range(len(Z))]
    yy = [Z[i] + 0.05 * xs[i] for i in range(len(Z))]
    r = SX.fit(yy, [[v] for v in xs], (0, 1, 1), (0, 1, 1), 12)
    assert abs(r["beta"][0] - 0.05) < 2.0 * r["beta_se"][0]
    assert r["beta_se"][0] > 0.0


def test_a_regressor_annihilated_by_differencing_is_refused():
    with pytest.raises(ValueError):
        SX.fit(Z, [[float(i)] for i in range(len(Z))], (0, 1, 1),
               (0, 1, 1), 12)


def test_a_constant_is_refused_when_d_plus_D_is_two():
    with pytest.raises(ValueError):
        SX.fit(Z, None, (0, 1, 1), (0, 1, 1), 12,
               include_constant=True)


def test_a_wrong_length_regressor_is_refused():
    with pytest.raises(ValueError):
        SX.fit(Z, [[1.0], [2.0]], (0, 1, 1), (0, 1, 1), 12)


def test_starting_models_are_the_papers_four():
    assert SX.starting_models(1, 1, 12) == [
        ((2, 1, 2), (1, 1, 1)), ((0, 1, 0), (0, 1, 0)),
        ((1, 1, 0), (1, 1, 0)), ((0, 1, 1), (0, 1, 1))]


def test_non_seasonal_starting_models_carry_no_seasonal_terms():
    assert all(so[0] == 0 and so[2] == 0
               for _, so in SX.starting_models(1, 0, 1))


def test_there_are_thirteen_neighbours():
    assert len(SX.neighbours((1, 1, 1), (1, 1, 1), False, 12)) == 13


def test_the_constant_switch_is_one_of_them():
    nb = SX.neighbours((1, 1, 1), (1, 1, 1), False, 12)
    assert ((1, 1, 1), (1, 1, 1), True) in nb


def test_the_upper_bounds_are_enforced():
    nb = SX.neighbours((5, 1, 5), (2, 1, 2), False, 12)
    assert all(o[0] <= 5 and o[2] <= 5 and so[0] <= 2 and so[2] <= 2
               for o, so, _ in nb)


def test_a_near_unit_root_is_rejected():
    assert not S._roots_ok([0.9995], SX.ROOT_TOL)
    assert S._roots_ok([0.5], SX.ROOT_TOL)


def test_aic_and_aicc():
    assert SX.aic(100.0, 3) == pytest.approx(-194.0)
    assert SX.aicc(100.0, 3, 50) > SX.aic(100.0, 3)
    assert SX.aicc(100.0, 3, 3) == float("inf")
