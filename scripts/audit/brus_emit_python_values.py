"""Emit Python-side values for the Brus-shelf R parity check."""

import json
import sys

import numpy as np

from morie.fn import _brus as br

out = {}
Z3 = [2.0, 5.0, 3.0]
PI3 = [0.5, 0.5, 0.5]

out["ht"] = {"total": br.ht_total(Z3, PI3), "mean": br.ht_mean(Z3, PI3, 10)}
out["si"] = {"p": br.si_proportion([1, 0, 1, 0]),
             "var_p": br.si_proportion_variance(0.5, 4, 20),
             "ci_lower": br.confidence_interval(10, 4, 1.645)["lower"],
             "total_inf": br.infinite_total(3.0, 100.0, 10.0),
             "var_total_inf": br.infinite_total_variance(2.0, 8, 100.0, 10.0)}
out["stsi"] = {"mean": br.stratified_mean([4.0, 6.0], [0.5, 0.5]),
               "var": br.stratified_variance([0.5, 0.8], [0.5, 0.5]),
               "cost": br.stratified_cost(10.0, [2.0, 3.0], [5, 4])}
out["cluster"] = {"pps": br.cluster_total_pps([7.0, 11.0], [2.0, 2.0], 6.0, 2),
                  "si": br.cluster_total_si([7.0, 11.0], 3, 2),
                  "mean": br.cluster_mean_from_total(27.0, 6.0),
                  "ts_mean": br.twostage_mean([3.5, 5.5, 4.5]),
                  "ts_var": br.twostage_variance_estimator(
                      [3.5, 5.5, 4.5])["variance"],
                  "true_var": br.twostage_variance_components(4.0, 9.0, 2, 3),
                  "total_si": br.twostage_total_si([10.0, 14.0], 5)}
out["design"] = {"m": br.twostage_optimal_m(3.0, 2.0, 10.0, 1.0),
                 "n_v": br.twostage_optimal_n_variance(3, 2, 10, 1, 0.5),
                 "n_b": br.twostage_optimal_n_budget(3, 2, 10, 1, 200.0)}
out["pps_var"] = br.pps_total_variance([2.0, 5.0], [0.2, 0.5], 10.0)
x4 = np.arange(1.0, 5)
z4 = np.arange(2.0, 10, 2)
out["ma"] = {
    "difference": br.difference_estimator(np.ones(6) * 4.0, Z3,
                                          np.ones(3) * 4.0, PI3, 6),
    "gls_b": float(br.gls_sample_slope(x4, z4, np.ones(4),
                                       np.full(4, 0.5))[0]),
    "regr_slopes": br.regression_estimator_slopes(3.0, [0.5], [4.0], [3.0]),
    "ratio": br.ratio_total(40.0, 20.0, 50.0),
    "ratio_g": br.ratio_g_weight(50.0, 20.0)}
e4 = [0.5, -0.3, 0.1, -0.3]
out["greg"] = {"s2_e": br.si_regression_variance(e4, 4, 20)["s2_e"],
               "var": br.si_regression_variance(e4, 4, 20)["variance"],
               "g_w": br.g_weight_simple(4.0, 5.0, 4.0, 2.0),
               "g_var": br.g_weighted_variance(np.ones(4), e4, 4, 20),
               "ratio_var": br.ratio_total_variance(e4, 4, 20),
               "mc_var": br.mc_variance_via_residuals(e4[:3], PI3, 6)}
out["calib"] = {"pst": br.poststratified_mean([4.0, 8.0], [0.25, 0.75]),
                "intercept": br.mixed_calibration_intercept(0.6, Z3, PI3, 6),
                "calibrated": br.mixed_calibration_mean(3.0, 0.4, PI3, 4.0,
                                                        3.5, 0.6, 6),
                "si": br.mixed_calibration_si(Z3, 0.6, 4.0, 3.5)}
out["bal"] = {"regr_total": br.regression_total(40.0, 50.0, 45.0, 0.8),
              "bal_var": br.balanced_variance([0.1, -0.2, 0.1], PI3,
                                              [1.0, 1.0, 1.0], 10, 1),
              "local_var": br.local_mean_variance([0.1, -0.2, 0.1], PI3,
                                                  [0.0, 0.0, 0.0], 3, 1),
              "tp_strat": br.twophase_stratified_variance(
                  [6, 4], 10, [1.0, 2.0], [3, 2], [4.0, 7.0], 5.2),
              "tp_regr": br.twophase_regression_variance(4.0, 10, 0.2, 3, 100),
              "s2_resid": br.s2_residuals([0.5, -0.5, 0.2], 3)}
out["nn"] = {"prop_se": br.n_for_proportion_se(0.5, 0.05),
             "mean_len": br.n_for_mean_length(1.96, 2.0, 1.0),
             "cv": br.n_for_cv(1.96, 0.4, 0.1),
             "prop_len": br.n_for_proportion_length(1.96, 0.5, 0.2),
             "de": br.n_design_effect(4.0, 50),
             "beta_pdf": br.beta_posterior_pdf(0.35, 7, 20, 1, 1),
             "interval": br.beta_posterior_interval_prob(0.2, 0.3, 7, 20,
                                                         1, 1),
             "alc": br.average_length_criterion([0.1, 0.2], [0.5, 0.5],
                                                0.2)["expected_length"],
             "acc": br.average_coverage_criterion([0.96, 0.95], [0.5, 0.5],
                                                  0.05)["expected_coverage"]}
out["ospats"] = {"stsi": br.mean_semivariance_stsi_variance(
                     [1.0, 2.0], [0.5, 0.5], [1.0, 1.0]),
                 "equal": br.mean_semivariance_equal_area([1.0, 2.0], 2),
                 "alloc": br.optimal_allocation_variance(
                     [0.6, 0.4], [3.0, 5.0], [1.0, 4.0], 10),
                 "obj_o": br.ospats_criterion_terms([0.6, 0.4], [3.0, 5.0]),
                 "d2": br.expected_squared_distance(3.0, 1.0, 4.0, 0.5,
                                                    0.7, 0.2),
                 "s2h": br.expected_stratum_variance(6.0, 3),
                 "objective": br.ospats_objective([4.0, 9.0], 10)}
s = np.array([0.0, 10.0, 20.0])
dm = np.abs(s[:, None] - s[None, :])
cov = np.where(dm == 0, 2.0, 2.0 * np.exp(-dm / 15.0))
cov0 = 2.0 * np.exp(-np.abs(s - 12.0) / 15.0)
sol = br.kriging_weights_covariance(cov, cov0)
out["krig"] = {"lam": sol["lam"].tolist(), "nu": sol["nu"],
               "v_cov": br.ok_variance_covariance_form(2.0, sol["lam"],
                                                       cov0, sol["nu"]),
               "v_gam": br.ok_variance_semivariance_form(
                   sol["lam"], 2.0 - cov0, -sol["nu"]),
               "gamma": br.exponential_semivariogram(75.0, 0.0, 2.0, 25.0),
               "loglik": br.gaussian_loglikelihood([0.0, 0.0], [0.0, 0.0],
                                                   np.eye(2))}
out["vgm"] = {"nested": br.nested_anova_prediction(1, 0.5, -0.2, 0.1, 0.05),
              "fisher": float(br.fisher_information_reml(
                  2.0 * np.eye(3), [np.eye(3)])[0, 0]),
              "vkv": br.variance_of_kriging_variance(
                  np.diag([0.04, 0.09]), [1.0, 2.0]),
              "akv": br.augmented_kriging_variance(1.5, 0.3),
              "tau2": br.expected_tau2(np.diag([0.04, 0.09]),
                                       [[1.0, 0.0], [0.0, 1.0]], np.eye(2)),
              "eac": br.estimation_adjusted_criterion(1.8, 1.5, 0.4)}
out["misc"] = {"sa": br.small_area_mb_mean([1.0, 2.0], [0.5, 0.25], 0.1),
               "tw0": float(br.trend_weights([1.0, 2.0, 3.0, 4.0])[0]),
               "gls0": float(br.gls_estimator(
                   np.column_stack([np.ones(3), np.arange(3.0)]),
                   np.eye(3), [1.0, 2.0, 3.0])[0]),
               "lm": br.linear_model_prediction(2.0, 3.0, 4.0),
               "ols0": float(br.ols_beta(
                   np.column_stack([np.ones(4), np.arange(4.0)]),
                   [1.0, 3.0, 5.0, 7.0])[0]),
               "pred_var": br.ols_prediction_variance(
                   2.0, [1.0, 2.0],
                   np.column_stack([np.ones(5), np.arange(5.0)])),
               "cls": br.classification_indicator("a", "a", "a"),
               "v_iid": br.iid_mean_variance(4.0, 8),
               "v_auto": br.autocorrelated_mean_variance(4.0, 8, 0.2),
               "n_eff": br.effective_sample_size(8, 0.2),
               "v_fpc": br.fpc_mean_variance(4.0, 8, 80)}

json.dump(out, open(sys.argv[1], "w"), indent=1)
n_vals = sum(len(v) if isinstance(v, dict) else 1 for v in out.values())
print("emitted", n_vals, "values")
