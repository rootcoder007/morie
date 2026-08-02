"""Emit Python-side values for the ACD-shelf R parity check."""

import json
import math
import sys

import numpy as np

from morie.fn import _acd as ac

out = {}
X8 = np.column_stack([np.ones(8), np.arange(8.0)])
Y8 = np.array([0.0, 0, 0, 1, 0, 1, 1, 1])


def wilson_fn(w, n):
    ci = ac.wilson_interval(w, n, 1.96)
    return ci["lower"], ci["upper"]


wi = ac.wilson_interval(4, 10, 1.96)
out["binom"] = {"pmf": ac.binomial_pmf(3, 10, 0.3),
                "var": ac.mle_variance_pi(0.4, 25),
                "wilson_est": wi["estimate"], "wilson_lo": wi["lower"],
                "true_level": ac.true_confidence_level(10, 0.3, wilson_fn)}
tg = ac.pearson_chi2_two_groups(12, 30, 20, 35)
orci = ac.or_wald_interval(20, 50, 10, 50, 1.96)
out["two_group"] = {"x2": tg["x2"],
                    "lrt": ac.lrt_two_groups(5, 20, 15, 20)["stat"],
                    "or": orci["or"], "or_lo": orci["lower"],
                    "or_hi": orci["upper"]}
fit = ac.logistic_mle(X8, Y8)
out["logistic"] = {"b0": float(fit["beta"][0]), "b1": float(fit["beta"][1]),
                   "loglik": fit["loglik"],
                   "cov00": float(fit["cov"][0, 0]),
                   "deviance": ac.residual_deviance(
                       1 / (1 + np.exp(-(X8 @ fit["beta"]))), Y8)}
orl = ac.or_ci_logistic(0.5, 0.04, 2.0, 1.96)
cov = np.array([[0.5, -0.1], [-0.1, 0.05]])
piw = ac.pi_wald_interval(0.2, 0.29, 1.96)
out["wald"] = {"or": orl["or"], "or_lo": orl["lower"],
               "var_xb": ac.linear_predictor_variance([1.0, 3.0], cov),
               "pi": piw["pi"], "pi_lo": piw["lower"]}
out["multinom"] = {"pmf": ac.multinomial_pmf([3, 7], [0.3, 0.7]),
                   "table": ac.contingency_pmf(
                       np.array([[2, 1], [1, 3]]),
                       np.array([[0.2, 0.1], [0.3, 0.4]])),
                   "product": ac.product_multinomial_pmf(
                       np.array([[2, 1], [1, 3]]),
                       np.array([[2 / 3, 1 / 3], [0.25, 0.75]]))}
probs = ac.baseline_probs([0.9, 0.1])
out["mlogit"] = {"logit": ac.baseline_logit(0.5, [0.3], [2.0]),
                 "p0": float(probs[0]), "p2": float(probs[2]),
                 "po": ac.proportional_odds_logit(0.5, [0.3], [2.0]),
                 "polr": ac.polr_parameterization(0.5, [-0.3], [2.0]),
                 "pi_j": ac.category_prob_from_cumulative(
                     [0.15, 0.45, 0.80], 2),
                 "wald_lo": ac.pi_j_wald_interval(0.3, 0.0025,
                                                  1.96)["lower"]}
sci = ac.poisson_score_interval(3.0, 20, 1.96)
out["poisson"] = {"score_lo": sci["lower"], "score_hi": sci["upper"],
                  "mu": ac.poisson_log_link(0.1, [1.0], [1.0]),
                  "loglik": ac.poisson_loglik(
                      [0.0, math.log(2)],
                      np.column_stack([np.ones(4), np.arange(4.0)]),
                      [1.0, 2, 4, 8]),
                  "mu_ind": ac.loglinear_independence_mean(1.0, 0.9, 0.4),
                  "mu_sat": ac.loglinear_saturated_mean(1.0, 0.9, 0.4, 0.7),
                  "or_ll": ac.loglinear_odds_ratio(0.0, 0.7, 0.0, 0.0),
                  "ratio": ac.ordinal_score_mean_ratio(0.4, 0.1, 0.2, 3.0,
                                                       1.0),
                  "rate": ac.poisson_rate_mean(0.1, [1.0], [1.0], 100.0)}
taus = ac.bic_posterior_probs([100.0, 102.0])
out["bic"] = {"tau0": float(taus[0]),
              "ma": ac.model_averaged_estimate([0.6, 0.4], [1.0, 2.0]),
              "var_ma": ac.model_averaged_variance([0.6, 0.4], [1.0, 2.0],
                                                   [0.1, 0.2])}
ec = ac.exact_conditional_pmf([0.0, 1.0, 2.0], [1, 4, 2], 0.5, 1.0)
kc = ac.kott_carr_interval(0.3, 0.01, 2.0)
out["extra"] = {"prev": ac.prevalence_from_apparent(0.1, 0.95, 0.98),
                "exact": ec["p_at_t"],
                "n_hat": ac.weighted_category_total([2.0, 3.0, 5.0],
                                                    ["a", "b", "a"], "a"),
                "jack": ac.jackknife_variance([1.0, 1.2, 0.8, 1.1], 1.0),
                "var_pi": ac.survey_proportion_variance(4.0, 9.0, 1.5,
                                                        0.3, 100.0),
                "kc_lo": kc["lower"], "kc_hi": kc["upper"],
                "spmi": ac.spmi_loglinear_mean(1.0, 0.2, 0.3),
                "three": ac.three_mrcv_mean(1.0, 0.2, 0.3, 0.1),
                "glmm": ac.glmm_linear_predictor(0.5, 2.0, 1.5, -0.2),
                "bayes_rule": ac.bayes_rule(0.99, 0.01, 0.05),
                "post_dens": ac.posterior_density_binomial(0.4, 7, 20, 1, 1),
                "bayes_est": ac.bayes_estimate_binomial(7, 20, 1, 1),
                "grid0": float(ac.posterior_kernel_regression(
                    [-3.0, -1.0, -2.0], [0.0, 0.0, 0.0])[0]),
                "et": ac.group_testing_expected_tests(5, 0.95, 0.98, 0.1),
                "gt_logit": ac.group_testing_logit(-1.0, [0.5], [2.0]),
                "piecewise": ac.piecewise_cubic(1.5, 2.0,
                                                [1.0, 0.5, -0.2, 0.1],
                                                [9.9, 9.9, 9.9, 9.9]),
                "spline": ac.truncated_power_spline(
                    3.0, [1.0, 0.5, -0.2, 0.1, 0.3], [2.0]),
                "spline_or": ac.spline_odds_ratio(
                    [1.0, 0.5, -0.2, 0.1, 0.3],
                    [lambda x: 1.0, lambda x: x, lambda x: x ** 2,
                     lambda x: x ** 3,
                     lambda x: (x - 2.0) ** 3 if x > 2.0 else 0.0],
                    3.0, 1.0)}

json.dump(out, open(sys.argv[1], "w"), indent=1)
print("emitted", sum(len(v) for v in out.values()), "values")
