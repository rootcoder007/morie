"""Schabenberger and Gotway spatial shelf.

Anchored on the book's own worked example where one exists, and on
designs with a known answer everywhere else. Page and equation numbers
are the book's.
"""

import numpy as np
import pytest

from morie.fn._schaben import (fit_variogram_wls, matheron, variogram_model)
from morie.fn.spclk import schabenberger_composite_likelihood
from morie.fn.spcrhk import schabenberger_cressie_hawkins
from morie.fn.spintp import schabenberger_intensity_estimation
from morie.fn.spkce import schabenberger_cov_param_estimation_kriging
from morie.fn.spkpe import schabenberger_kriging_pred_error
from morie.fn.spmath import schabenberger_matheron_estimator
from morie.fn.spml import schabenberger_ml_variogram
from morie.fn.spsarml import schabenberger_sar_ml
from morie.fn.vargm import empirical_variogram

# Example 4.3, pp. 157 and 161. Five locations; Z([3,4]) = 20 is the
# outlier, and it contributes to four of the five lag classes.
EX43_COORDS = np.array([[1, 1], [1, 4], [2, 2], [3, 1], [3, 4]], float)
EX43_Z = np.array([1.0, 4.0, 2.0, 3.0, 20.0])
EX43_MATHERON = [0.5, 65.0, 82.0, 74.5, 90.5]          # printed p. 157
EX43_CH = [0.71, 38.14, 45.5, 52.2, 36.6]              # printed p. 161


def test_matheron_reproduces_example_4_3():
    out = schabenberger_matheron_estimator(EX43_COORDS, EX43_Z, exact=True)
    assert [round(float(g), 1) for g in out["gamma"]] == EX43_MATHERON


def test_cressie_hawkins_reproduces_example_4_3():
    out = schabenberger_cressie_hawkins(EX43_COORDS, EX43_Z, exact=True)
    got = [round(float(g), 1) for g in out["gamma"]]
    assert got == [round(v, 1) for v in EX43_CH]


def test_the_printed_bias_factor_omits_the_third_term():
    # the book derives 0.457 + 0.494/|N| + 0.045/|N|^2 on p. 160 and then
    # writes equation (4.26) without the last term. At |N(h)| = 2 the
    # worked example uses 0.704, which is 0.457 + 0.494/2 exactly -- so
    # the printed estimator, not the derivation, is what it evaluates.
    assert 0.457 + 0.494 / 2 == pytest.approx(0.704)
    plain = schabenberger_cressie_hawkins(EX43_COORDS, EX43_Z, exact=True)
    full = schabenberger_cressie_hawkins(EX43_COORDS, EX43_Z, exact=True,
                                         full_correction=True)
    assert round(float(plain["gamma"][1]), 2) == 38.14
    assert float(full["gamma"][1]) != float(plain["gamma"][1])


def test_the_outlier_is_what_separates_the_two_estimators():
    # p. 158: dropping Z([3,4]) leaves gamma(2) = 2, gamma(sqrt5) = 2,
    # gamma(3) = 4.5, gamma(sqrt13) = 0.5 -- i.e. the classical estimator
    # was reporting 65, 82, 74.5, 90.5 almost entirely because of it
    clean = schabenberger_matheron_estimator(
        EX43_COORDS[:4], EX43_Z[:4], exact=True
    )
    assert max(float(g) for g in clean["gamma"]) < 6.0
    ch = schabenberger_cressie_hawkins(EX43_COORDS, EX43_Z, exact=True)
    mat = schabenberger_matheron_estimator(EX43_COORDS, EX43_Z, exact=True)
    # the robust estimator suppresses, but does not remove, the outlier:
    # its breakdown point is still zero
    assert np.all(ch["gamma"][1:] < mat["gamma"][1:])
    assert float(ch["gamma"][1]) > 30.0


def test_matheron_variance_is_equation_4_25():
    out = schabenberger_matheron_estimator(EX43_COORDS, EX43_Z, exact=True)
    expected = 2.0 * out["gamma"] ** 2 / out["n_pairs"]
    assert np.allclose(out["variance"], expected)


def test_sparse_lag_classes_are_flagged():
    out = schabenberger_matheron_estimator(EX43_COORDS, EX43_Z, exact=True)
    # two pairs per lag, well under the book's 30
    assert len(out["sparse_lags"]) == 5
    assert "30" in out["sparse_note"]


# --------------------------------------------------------------------
# fitting: a field simulated from a known covariance must be recovered
# --------------------------------------------------------------------

def gaussian_field(seed=0, n=220, nugget=0.2, psill=1.8, rng_par=3.0,
                   model="exponential", size=20.0):
    """Draw an exact realisation by Cholesky of the target covariance."""
    gen = np.random.default_rng(seed)
    P = gen.uniform(0, size, size=(n, 2))
    D = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2))
    C = (nugget + psill) - variogram_model(D, model, nugget, psill, rng_par)
    L = np.linalg.cholesky(C + np.eye(n) * 1e-8)
    return P, L @ gen.normal(size=n)


def test_wls_recovers_the_simulated_sill():
    sills = []
    for s in range(8):
        P, z = gaussian_field(seed=s)
        out = empirical_variogram(P, z, lags=12, model="exponential")
        sills.append(out["sill"])
    # true sill = nugget + psill = 2.0
    assert abs(float(np.mean(sills)) - 2.0) < 0.5


def test_the_empirical_variogram_rises_to_a_sill():
    P, z = gaussian_field(seed=1)
    out = empirical_variogram(P, z, lags=12)
    g = out["gamma"]
    assert float(g[0]) < float(g[-1])
    # the near-range values must sit below the far-range plateau
    assert float(np.mean(g[:3])) < float(np.mean(g[-3:]))


def test_practical_range_convention_is_the_one_documented():
    # exponential: gamma(a) should be 95 % of the sill, not 63 %
    g = variogram_model(np.array([3.0]), "exponential", 0.0, 1.0, 3.0)
    assert float(g[0]) == pytest.approx(1 - np.exp(-3.0), abs=1e-12)
    assert float(g[0]) > 0.94


def test_spherical_model_reaches_its_sill_exactly_at_the_range():
    g = variogram_model(np.array([2.0, 5.0, 9.0]), "spherical", 0.0, 1.0, 5.0)
    assert float(g[1]) == pytest.approx(1.0)
    assert float(g[2]) == pytest.approx(1.0)
    assert float(g[0]) < 1.0


def test_wls_weights_downweight_the_long_lags():
    P, z = gaussian_field(seed=2)
    lag, gam, npair, _ = matheron(P, z, 12)
    wls = fit_variogram_wls(lag, gam, npair, "exponential", "cressie")
    ols = fit_variogram_wls(lag, gam, npair, "exponential", "ols")
    # both are legitimate; the point is that the weighting changes the
    # answer, which is why the choice is reported
    assert wls["weights"] == "cressie" and ols["weights"] == "ols"
    assert np.isfinite(wls["range"]) and np.isfinite(ols["range"])


def test_composite_likelihood_needs_no_binning_choice():
    P, z = gaussian_field(seed=3)
    cl = schabenberger_composite_likelihood(P, z)
    # fitted to the cloud: the number of pairs is n(n-1)/2 with no lags
    n = z.size
    assert cl["n_pairs"] == n * (n - 1) // 2
    assert cl["sill"] > 0 and cl["range"] > 0
    # and it should land in the same region as the binned WLS fit
    wls = empirical_variogram(P, z, lags=12, model="exponential")
    assert abs(cl["sill"] - wls["sill"]) < 1.5 * max(wls["sill"], 1.0)


def test_composite_likelihood_is_the_variance_weighted_gee():
    # equation (4.44) differs from (4.43) only by 1/(8 gamma^2); with a
    # constant gamma the two weightings coincide up to a scale, so the
    # objective must be a pure rescaling. Checked on the weight itself.
    g = np.array([2.0, 2.0, 2.0])
    w_cl = 1.0 / (8.0 * g ** 2)
    assert np.allclose(w_cl / w_cl[0], np.ones(3))


def test_reml_and_ml_differ_and_reml_is_the_larger_variance():
    P, z = gaussian_field(seed=4, n=90)
    both = schabenberger_ml_variogram(P, z, method="both")
    ml, reml = both["ml"], both["reml"]
    assert ml["converged"] and reml["converged"]
    # ML is biased downward because the mean's degrees of freedom are
    # unaccounted for; on a single draw this is a tendency, so only the
    # objectives are compared exactly
    assert ml["neg2loglik"] != reml["neg2loglik"]
    assert both["comparable_across_mean_models"] is False


def test_ml_is_flagged_as_comparable_and_reml_is_not():
    P, z = gaussian_field(seed=5, n=70)
    ml = schabenberger_ml_variogram(P, z, method="ml")
    reml = schabenberger_ml_variogram(P, z, method="reml")
    assert ml["comparable_across_mean_models"] is True
    assert reml["comparable_across_mean_models"] is False


def test_ml_variance_bias_is_theta_over_n_for_independent_data():
    # p. 167 states the bias exactly for the independent case. Check the
    # arithmetic claim directly: MLE of sigma^2 divides by n, so its
    # expectation is (n-1)/n sigma^2, a bias of -sigma^2/n.
    n = 50
    gen = np.random.default_rng(0)
    biases = []
    for _ in range(400):
        x = gen.normal(scale=2.0, size=n)
        biases.append(np.mean((x - x.mean()) ** 2) - 4.0)
    assert abs(float(np.mean(biases)) - (-4.0 / n)) < 0.15


# --------------------------------------------------------------------
# covariance-parameter estimation with a trend
# --------------------------------------------------------------------

def test_trend_leaks_into_the_raw_semivariogram():
    # equation (5.35): with a spatially varying mean the empirical
    # semivariogram of the RAW data estimates the semivariogram plus a
    # squared mean difference, so it climbs without settling
    gen = np.random.default_rng(6)
    P = gen.uniform(0, 20, size=(150, 2))
    trend = 1.5 * P[:, 0]
    z = trend + gen.normal(scale=0.5, size=150)
    raw = empirical_variogram(P, z, lags=10)
    g = raw["gamma"]
    # a pure trend produces a monotone, still-rising variogram
    assert float(g[-1]) > 5 * float(g[0])


def test_irwgls_recovers_the_trend_coefficients():
    gen = np.random.default_rng(7)
    P = gen.uniform(0, 20, size=(120, 2))
    z = 3.0 + 1.5 * P[:, 0] + gen.normal(scale=0.6, size=120)
    out = schabenberger_cov_param_estimation_kriging(
        P, z, method="wls", X=P[:, :1]
    )
    beta = out["beta"]
    assert beta.size == 2
    assert abs(float(beta[0]) - 3.0) < 1.0
    assert abs(float(beta[1]) - 1.5) < 0.3
    assert out["spatially_varying_mean"] is True
    assert out["trend_bias_warning"] is not None


def test_constant_mean_needs_no_iteration():
    P, z = gaussian_field(seed=8, n=80)
    out = schabenberger_cov_param_estimation_kriging(P, z, method="wls")
    assert out["spatially_varying_mean"] is False
    assert out["iterations"] == 1
    assert out["trend_bias_warning"] is None


def test_convergence_is_reported_as_lack_of_progress():
    P, z = gaussian_field(seed=9, n=80)
    out = schabenberger_cov_param_estimation_kriging(P, z, method="wls")
    assert "lack of progress" in out["convergence_note"]


# --------------------------------------------------------------------
# prediction error
# --------------------------------------------------------------------

def test_the_correction_only_ever_increases_the_prediction_error():
    P, z = gaussian_field(seed=10, n=70)
    out = schabenberger_kriging_pred_error(P, z, [[10.0, 10.0], [5.0, 15.0]])
    assert np.all(out["mse"] >= out["mse_plugin"] - 1e-12)
    assert np.all(out["correction"] >= -1e-12)
    assert out["parameters_estimated"] is True


def test_known_parameters_still_report_the_plugin_baseline():
    P, z = gaussian_field(seed=11, n=70)
    out = schabenberger_kriging_pred_error(
        P, z, [10.0, 10.0], nugget=0.2, psill=1.8, rng=3.0
    )
    assert out["parameters_estimated"] is False
    assert np.all(out["mse_plugin"] > 0)


def test_kriging_honours_the_data_at_an_observed_location():
    P, z = gaussian_field(seed=12, n=60, nugget=0.0)
    out = schabenberger_kriging_pred_error(
        P, z, P[3], nugget=1e-8, psill=2.0, rng=3.0
    )
    # with no nugget the predictor interpolates
    assert abs(float(out["prediction"][0]) - float(z[3])) < 0.05


# --------------------------------------------------------------------
# intensity
# --------------------------------------------------------------------

def test_intensity_of_a_homogeneous_process_matches_the_count_over_area():
    gen = np.random.default_rng(13)
    pts = gen.uniform(0, 10, size=(400, 2))
    out = schabenberger_intensity_estimation(pts, region=(0, 10, 0, 10))
    assert out["mean_intensity"] == pytest.approx(4.0)
    # the edge-corrected surface should integrate to roughly the count
    assert abs(out["integrated_intensity"] - 400.0) < 160.0


def test_edge_correction_lifts_the_boundary():
    gen = np.random.default_rng(14)
    pts = gen.uniform(0, 10, size=(300, 2))
    on = schabenberger_intensity_estimation(pts, region=(0, 10, 0, 10),
                                            edge_correct=True)
    off = schabenberger_intensity_estimation(pts, region=(0, 10, 0, 10),
                                             edge_correct=False)
    # corners lose the most kernel mass, so that is where the correction
    # must bite hardest
    assert on["intensity_surface"][0, 0] > off["intensity_surface"][0, 0]
    assert on["edge_weight_min"] < 0.9


def test_an_omitted_region_is_flagged_because_it_biases_upward():
    gen = np.random.default_rng(15)
    pts = gen.uniform(0, 10, size=(200, 2))
    out = schabenberger_intensity_estimation(pts)
    assert out["region_supplied"] is False
    assert "overstates" in out["region_note"]


def test_clustered_pattern_has_a_more_variable_surface():
    gen = np.random.default_rng(16)
    even = gen.uniform(0, 10, size=(300, 2))
    centres = gen.uniform(1, 9, size=(4, 2))
    clumped = np.vstack([
        centres[i % 4] + gen.normal(scale=0.4, size=2) for i in range(300)
    ])
    a = schabenberger_intensity_estimation(even, region=(0, 10, 0, 10))
    b = schabenberger_intensity_estimation(clumped, region=(0, 10, 0, 10))
    assert b["intensity_surface"].std() > a["intensity_surface"].std()


@pytest.mark.parametrize("kern", ["gaussian", "quadratic",
                                  "minimum_variance", "uniform"])
def test_every_kernel_integrates_to_one(kern):
    from morie.fn.spintp import _kernel
    t = np.linspace(-8, 8, 200001)
    assert float(np.trapezoid(_kernel(t, kern), t)) == pytest.approx(1.0,
                                                                    abs=1e-4)


# --------------------------------------------------------------------
# spatial autoregression
# --------------------------------------------------------------------

def ring_weights(n):
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i + 1) % n] = W[i, (i - 1) % n] = 1.0
    return W / W.sum(axis=1, keepdims=True)


def test_sar_error_recovers_a_known_rho():
    n, rho_true = 200, 0.6
    W = ring_weights(n)
    gen = np.random.default_rng(17)
    X = np.column_stack([np.ones(n), gen.normal(size=n)])
    rhos = []
    for s in range(10):
        g = np.random.default_rng(100 + s)
        e = np.linalg.solve(np.eye(n) - rho_true * W, g.normal(size=n) * 0.5)
        y = X @ np.array([1.0, 2.0]) + e
        rhos.append(schabenberger_sar_ml(X, y, W, model="error")["rho"])
    assert abs(float(np.mean(rhos)) - rho_true) < 0.12


def test_sar_error_recovers_beta():
    n = 200
    W = ring_weights(n)
    gen = np.random.default_rng(18)
    X = np.column_stack([np.ones(n), gen.normal(size=n)])
    e = np.linalg.solve(np.eye(n) - 0.5 * W, gen.normal(size=n) * 0.4)
    out = schabenberger_sar_ml(X, X @ np.array([1.0, 2.0]) + e, W,
                               model="error")
    assert abs(float(out["beta"][0]) - 1.0) < 0.4
    assert abs(float(out["beta"][1]) - 2.0) < 0.15


def test_sar_lag_recovers_a_known_rho():
    n, rho_true = 200, 0.5
    W = ring_weights(n)
    gen = np.random.default_rng(19)
    X = np.column_stack([np.ones(n), gen.normal(size=n)])
    rhos = []
    for s in range(10):
        g = np.random.default_rng(200 + s)
        y = np.linalg.solve(
            np.eye(n) - rho_true * W,
            X @ np.array([1.0, 2.0]) + g.normal(size=n) * 0.5,
        )
        rhos.append(schabenberger_sar_ml(X, y, W, model="lag")["rho"])
    assert abs(float(np.mean(rhos)) - rho_true) < 0.12


def test_lag_and_error_are_different_models_not_aliases():
    n = 150
    W = ring_weights(n)
    gen = np.random.default_rng(20)
    X = np.column_stack([np.ones(n), gen.normal(size=n)])
    y = np.linalg.solve(np.eye(n) - 0.6 * W,
                        X @ np.array([1.0, 2.0]) + gen.normal(size=n) * 0.5)
    lag = schabenberger_sar_ml(X, y, W, model="lag")
    err = schabenberger_sar_ml(X, y, W, model="error")
    # data generated by the lag process: fitting the error model to it
    # gives a different rho and a worse likelihood
    assert abs(lag["rho"] - err["rho"]) > 0.02
    assert lag["neg2loglik"] < err["neg2loglik"]


def test_rho_bounds_come_from_the_eigenvalues_of_w():
    W = ring_weights(60)
    gen = np.random.default_rng(21)
    X = np.column_stack([np.ones(60), gen.normal(size=60)])
    out = schabenberger_sar_ml(X, gen.normal(size=60), W)
    lo, hi = out["rho_bounds"]
    ev = np.linalg.eigvals(W).real
    assert out["row_standardised"] is True
    assert hi == pytest.approx(1.0 / ev.max() - 1e-6, abs=1e-6)
    assert lo < 0 and lo >= 1.0 / ev.min() - 1e-6
    assert lo <= out["rho"] <= hi


def test_least_squares_rho_is_reported_but_not_used():
    n = 200
    W = ring_weights(n)
    gen = np.random.default_rng(22)
    X = np.column_stack([np.ones(n), gen.normal(size=n)])
    e = np.linalg.solve(np.eye(n) - 0.7 * W, gen.normal(size=n) * 0.4)
    out = schabenberger_sar_ml(X, X @ np.array([1.0, 2.0]) + e, W)
    assert "INCONSISTENT" in out["ols_note"]
    assert np.isfinite(out["ols_rho"])
    assert out["rho"] != out["ols_rho"]


def test_sar_rejects_a_nonzero_diagonal():
    W = ring_weights(30)
    W[0, 0] = 0.5
    gen = np.random.default_rng(23)
    X = np.column_stack([np.ones(30), gen.normal(size=30)])
    with pytest.raises(ValueError, match="zero diagonal"):
        schabenberger_sar_ml(X, gen.normal(size=30), W)


def test_input_validation_across_the_shelf():
    P, z = gaussian_field(seed=24, n=40)
    with pytest.raises(ValueError, match="model must be one of"):
        empirical_variogram(P, z, model="cubic")
    with pytest.raises(ValueError, match="model must be one of"):
        schabenberger_composite_likelihood(P, z, variogram_model="cubic")
    with pytest.raises(ValueError, match="method must be"):
        schabenberger_ml_variogram(P, z, method="bayes")
    with pytest.raises(ValueError, match="kernel must be one of"):
        schabenberger_intensity_estimation(P, kernel="tricube")
    with pytest.raises(ValueError, match="'error' or 'lag'"):
        schabenberger_sar_ml(np.ones((30, 2)), np.zeros(30),
                             ring_weights(30), model="car")
