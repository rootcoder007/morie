"""Generalised linear models, anchored on R's glm.

Every literal is a summary(glm(...)) value from the same 40-observation
fixture, at full precision.
"""
import math

import pytest

from morie.fn import _glm_core as G

N = 40
X1 = [((i * 7) % 11) / 5 - 1 for i in range(N)]
X2 = [((i * 5) % 7) / 3 - 1 for i in range(N)]
X = [[X1[i], X2[i]] for i in range(N)]
YB = [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1,
      1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
YC = [2, 0, 3, 1, 4, 2, 1, 0, 5, 3, 2, 1, 6, 2, 0, 3, 4, 1, 2, 5,
      1, 3, 0, 2, 4, 2, 1, 3, 2, 6, 0, 2, 3, 1, 4, 2, 5, 1, 3, 2]
YG = [1.2, 3.4, 0.8, 2.1, 5.6, 1.9, 2.7, 0.5, 4.3, 3.1,
      1.4, 2.2, 6.1, 1.7, 0.9, 3.3, 4.8, 1.1, 2.5, 5.2,
      1.6, 3.7, 0.7, 2.4, 4.1, 1.8, 2.9, 3.6, 2.0, 6.4,
      0.6, 2.3, 3.9, 1.3, 4.5, 2.6, 5.1, 1.5, 3.2, 2.8]


def test_logistic_matches_glm_binomial():
    f = G.glm(YB, X, "binomial")
    assert abs(f["coef"][0] - 0.575508023961936) < 1e-12
    assert abs(f["coef"][1] - 0.674299692288504) < 1e-12
    assert abs(f["coef"][2] - (-0.892689060687477)) < 1e-12
    assert abs(f["se"][0] - 0.352865639103213) < 1e-12
    assert abs(f["se"][1] - 0.554161354747213) < 1e-12
    assert abs(f["se"][2] - 0.531199272470709) < 1e-12
    assert f["converged"]


def test_logistic_wald_statistics_match_summary_glm():
    f = G.glm(YB, X, "binomial")
    assert f["statistic_name"] == "z"
    assert abs(f["statistic"][1] - 1.21679306308917) < 1e-12
    assert abs(f["statistic"][2] - (-1.68051634659703)) < 1e-12
    assert abs(f["p_value"][1] - 0.223682960012271) < 1e-12
    assert abs(f["p_value"][2] - 0.0928568965556806) < 1e-12
    # binomial dispersion is fixed at 1, so the statistic is normal
    assert f["dispersion"] == 1.0


def test_logistic_deviance_and_aic_match_glm():
    f = G.glm(YB, X, "binomial")
    assert abs(f["deviance"] - 48.1888471033342) < 1e-12
    assert abs(f["null_deviance"] - 52.9250590526386) < 1e-12
    assert abs(f["aic"] - 54.1888471033342) < 1e-12
    assert f["df_residual"] == 37
    assert f["df_null"] == 39


def test_standard_errors_use_the_final_solve_weights():
    # summary.glm inverts the QR stored by the last IRLS step, whose
    # weights sit at the PREVIOUS eta.  Recomputing them at the
    # converged eta shifts every standard error in its eighth digit --
    # invisible in beta, visible here, which is why this is pinned.
    f = G.glm(YB, X, "binomial")
    assert abs(f["se"][0] - 0.352865639103213) < 1e-12
    assert abs(f["se"][0] - 0.352865645205766) > 1e-10


def test_fitted_values_and_deviance_residuals_match_glm():
    f = G.glm(YB, X, "binomial")
    assert abs(f["fitted"][0] - 0.68866756307354) < 1e-12
    assert abs(f["fitted"][9] - 0.727125645237499) < 1e-12
    dr = G.deviance_residuals(f, YB)
    assert abs(dr[0] - (-1.52767405468362)) < 1e-12
    assert abs(dr[1] - 1.07321463302608) < 1e-12
    # their sum of squares IS the deviance -- that is what makes them
    # the right residual to plot for a non-normal family
    assert abs(sum(d * d for d in dr) - f["deviance"]) < 1e-10


def test_poisson_matches_glm_poisson():
    f = G.glm(YC, X, "poisson")
    assert abs(f["coef"][0] - 0.840912082217813) < 1e-12
    assert abs(f["coef"][1] - (-0.0653858891755061)) < 1e-12
    assert abs(f["coef"][2] - 0.236522669977984) < 1e-12
    assert abs(f["se"][0] - 0.104553588511299) < 1e-12
    assert abs(f["se"][1] - 0.160148592152359) < 1e-12
    assert abs(f["p_value"][2] - 0.124039071766556) < 1e-12
    assert abs(f["deviance"] - 49.0196717101663) < 1e-12
    assert abs(f["null_deviance"] - 51.6568945656903) < 1e-12
    assert abs(f["aic"] - 151.532951940163) < 1e-11


def test_offset_shifts_only_the_intercept_of_a_log_link():
    # a constant log-offset is absorbed by the intercept and must leave
    # every slope untouched -- the property that makes rate models work
    f = G.glm(YC, X, "poisson")
    o = G.glm(YC, X, "poisson", offset=[math.log(2)] * N)
    assert abs(o["coef"][0] - 0.147764901657868) < 1e-12
    assert abs(o["coef"][1] - (-0.0653858891755061)) < 1e-12
    assert abs(o["coef"][0] - (f["coef"][0] - math.log(2))) < 1e-10
    assert abs(o["coef"][2] - f["coef"][2]) < 1e-10


def test_gaussian_glm_is_least_squares():
    f = G.glm(YG, X, "gaussian")
    assert abs(f["coef"][0] - 2.79318065495427) < 1e-12
    assert abs(f["coef"][1] - 0.181934504572747) < 1e-12
    assert abs(f["coef"][2] - 0.825089714755335) < 1e-12
    assert abs(f["se"][1] - 0.371965530972281) < 1e-12
    assert abs(f["deviance"] - 84.3547827104239) < 1e-12
    # deviance IS the residual sum of squares for this family
    assert abs(f["deviance"]
               - sum(r * r for r in f["residuals"])) < 1e-10


def test_gaussian_dispersion_is_estimated_so_the_statistic_is_t():
    f = G.glm(YG, X, "gaussian")
    assert f["statistic_name"] == "t"
    assert abs(f["dispersion"] - 2.27985899217362) < 1e-12
    assert abs(f["statistic"][1] - 0.489116569745558) < 1e-12
    assert abs(f["p_value"][1] - 0.627646213372388) < 1e-12
    assert abs(f["aic"] - 151.361164818223) < 1e-10


def test_gamma_log_link_matches_glm_Gamma():
    f = G.glm(YG, X, "gamma")
    assert abs(f["coef"][0] - 1.0064775735195) < 1e-12
    assert abs(f["coef"][1] - 0.037055477172802) < 1e-12
    assert abs(f["coef"][2] - 0.306389256569374) < 1e-12
    assert abs(f["se"][1] - 0.133798105402901) < 1e-12
    assert abs(f["se"][2] - 0.126763693115854) < 1e-12
    assert abs(f["dispersion"] - 0.294986758013349) < 1e-12
    assert abs(f["deviance"] - 12.9790508743701) < 1e-9


def test_predict_link_and_response_are_the_two_scales():
    f = G.glm(YB, X, "binomial")
    eta = G.glm_predict(f, X, type="link")
    mu = G.glm_predict(f, X, type="response")
    assert abs(eta[0] - f["linear_predictor"][0]) < 1e-12
    assert abs(mu[0] - f["fitted"][0]) < 1e-12
    # the link scale is unbounded, the response scale never leaves (0,1)
    assert all(0.0 < m < 1.0 for m in mu)
    for e, m in zip(eta, mu):
        assert abs(m - 1.0 / (1.0 + math.exp(-e))) < 1e-12


def test_prior_weights_replicate_duplicated_observations():
    # weight 2 on every row must give the same coefficients as stacking
    # the data twice, which is the definition of a prior weight
    a = G.glm(YB, X, "binomial", weights=[2.0] * N)
    b = G.glm(YB + YB, X + X, "binomial")
    for u, v in zip(a["coef"], b["coef"]):
        assert abs(u - v) < 1e-9
    assert abs(a["deviance"] - b["deviance"]) < 1e-8


def test_no_intercept_fit_drops_the_column():
    f = G.glm(YC, X, "poisson", add_intercept=False)
    assert f["k"] == 2
    assert f["df_residual"] == 38
    assert f["df_null"] == 40


def test_glm_inputs_are_validated():
    with pytest.raises(ValueError):
        G.glm(YB, X, "binomial2")
    with pytest.raises(ValueError):
        G.glm(YB[:5], X, "binomial")
    with pytest.raises(ValueError):
        G.glm([2.0] * N, X, "binomial")          # outside [0, 1]
    with pytest.raises(ValueError):
        G.glm([-1.0] * N, X, "poisson")          # negative count
    with pytest.raises(ValueError):
        G.glm(YB[:2], X[:2], "binomial")         # n <= p


def test_collinear_predictors_are_refused_not_silently_fudged():
    Xc = [[X1[i], 2.0 * X1[i]] for i in range(N)]
    with pytest.raises(ValueError, match="collinear"):
        G.glm(YB, Xc, "binomial")


def test_predict_rejects_a_mismatched_design():
    f = G.glm(YB, X, "binomial")
    with pytest.raises(ValueError):
        G.glm_predict(f, [[1.0]] * 3)
    with pytest.raises(ValueError):
        G.glm_predict(f, X, type="probability")
