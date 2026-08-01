"""Emit Python-side values for the Criminology-shelf R parity check."""

import json
import sys

import numpy as np

from morie.fn import _ca_crim as ca

X21 = [1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5]
Y21 = [1, 0, 1, 2, 2, 0, 1, 1, 3, 3, 3, 4, 2, 4, 4, 4, 5, 6, 9, 7]

out = {}

fit = ca.ols_simple(X21, Y21)
out["ols_simple"] = {k: fit[k] for k in ("b1", "b0", "r", "se_b1", "t")}
out["ols_two_iv"] = ca.ols_two_iv(0.7156, 0.7616, 0.6280, 2.300, 0.9631, 2.658)
yhat = [fit["b0"] + fit["b1"] * x for x in X21]
vp = ca.variance_partition(Y21, yhat)
out["fit_indices"] = {"var_total": vp["var_total"], "var_model": vp["var_model"],
                      "var_resid": vp["var_resid"], "r2": vp["r2"],
                      "adj_r2": ca.adjusted_r2(vp["r2"], 20, 1),
                      "f_from_r2": ca.f_overall_r2(vp["r2"], 20, 1)}
out["f_change"] = {"f_ss": ca.f_nested_ss(80.0, 60.0, 5, 2, 100)["f"],
                   "f_r2": ca.f_nested_r2(0.4, 0.3, 5, 2, 100)["f"]}
out["std_coef"] = {"ols": ca.beta_standardized(2.0, 1.5, 3.0),
                   "logistic": ca.beta_logistic(0.5, 0.4),
                   "gelman": ca.beta_logistic(0.5, 0.4, gelman=True)}
out["vif_tol"] = {"tolerance": ca.tolerance(0.06), "vif": ca.vif(0.06)}
out["logit_link"] = {"logit": ca.logit(0.3), "odds": ca.odds(0.3),
                     "p_from_xb": ca.inv_logit(0.205),
                     "odds_ratio": ca.odds_ratio_unit_change(0.805)}
out["logistic_effects"] = {
    "dm": ca.derivative_at_mean(0.5, 0.8),
    "pct": ca.percent_correct_predictions(80, 100),
    "chi2": ca.model_chi2(528.171, 492.513),
    "wald": ca.wald_statistic(0.805, 0.332),
    "cox_snell": ca.cox_snell_r2(528.171, 492.513, 417),
    "lr": ca.likelihood_ratio_chi2(499.447, 492.513)}
out["mlogit"] = {"probs": ca.multinomial_probs([0.0, 1.2, -0.4]).tolist(),
                 "cond_or": ca.multinomial_conditional_or(1.2, -0.4)}
out["ordinal"] = {"cum_prob": ca.cumulative_probability([.15, .30, .35, .20], 2),
                  "cum_logit": ca.cumulative_logit([.15, .30, .35, .20], 1),
                  "plus": ca.ordinal_logit(0.5, [0.3], [2.0], "plus"),
                  "minus": ca.ordinal_logit(0.5, [0.3], [2.0], "minus")}
yq = [2.0, 0.0, 3.0, 1.0, 4.0, 2.0]
yhq = [1.5, 0.8, 2.5, 1.2, 3.5, 2.0]
out["count_glm"] = {"predict": ca.poisson_loglink_predict(-1.0, 0.736, 2.0),
                    "irr": ca.incidence_rate_ratio(0.736),
                    "offset": ca.poisson_offset_predict(-1.0, 0.736, 2.0, 100.0),
                    "theta": ca.quasi_poisson_theta(yq, yhq, 1),
                    "se_quasi": ca.quasi_poisson_se(
                        0.083, ca.quasi_poisson_theta(yq, yhq, 1)),
                    "negbin_var": ca.negative_binomial_variance(2.0, 0.5)}
out["hlm"] = {"sigma2_u": ca.variance_components_sigma2_u(3.9096, 0.27, 117.41),
              "icc": ca.intraclass_correlation(0.031, 0.270),
              "lr_chi2": ca.lr_test_chi2(-1871.73, -1777.35)}
pw = ca.power_from_delta_t(ca.noncentrality_delta_d(0.2, 100, 100),
                           1.6526, 198)
out["power"] = {"delta_d": ca.noncentrality_delta_d(0.2, 100, 100),
                "t_beta": pw["t_beta"], "beta": pw["beta"], "power": pw["power"],
                "lambda": ca.noncentrality_lambda_f(0.25, 300),
                "delta_r": ca.noncentrality_delta_r(0.3, 100),
                "r2_f2": ca.r2_from_f2(0.0625)}
ti = ca.t_independent(127.8, 132.3, 10.4, 9.8, 25, 30)
out["rct"] = {"b_t": ca.treatment_b_confounded(-0.25, -0.50, 0.50, 1.0, 1.0),
              "b_t_random": ca.treatment_b_randomized(0.3, 2.0, 1.0),
              "t": ti["t"], "s_pooled": ti["s_pooled"],
              "chi2": ca.chi2_2x2(30, 20, 15, 35)["chi2"],
              "t_paired": ca.t_paired([1.0, 2.0, 0.5, 1.5, 1.0])["t"]}
g3 = [[1.0, 2, 3, 2.5], [4.0, 5, 6, 5.5], [7.0, 8, 9, 8.5]]
an = ca.anova_oneway(g3)
out["anova"] = {"ms_between": an["ms_between"], "ms_within": an["ms_within"],
                "f": an["f"]}
yb = [1.0, 2.1, 6.2, 6.9, 11.1, 12.0, 15.8, 17.1]
bl = ca.anova_randomized_block(yb, [0, 1, 0, 1, 0, 1, 0, 1],
                               [0, 0, 1, 1, 2, 2, 3, 3])
out["block"] = {"f_treatment": bl["f_treatment"], "ss_block": bl["ss_block"]}
out["psm"] = {"bias": ca.psm_standardized_bias(0.5, 0.4, 0.2, 0.2)}
d = ca.cohens_d_sample(127.8, 132.3, 10.4, 9.8, 25, 30)
g = ca.hedges_g(d, 25, 30)
out["meta_es"] = {"d": d, "s_pooled": ca.pooled_sd(10.4, 9.8, 25, 30),
                  "j": ca.hedges_j(25, 30), "g": g, "se_g": ca.se_g(g, 25, 30),
                  "d_from_t": ca.d_from_t(ti["t"], 25, 30),
                  "rr": ca.risk_ratio(40, 60, 55, 45),
                  "or": ca.odds_ratio_2x2(40, 60, 55, 45),
                  "se_ln_rr": ca.se_log_rr(0.4, 0.55, 100, 100),
                  "se_ln_or": ca.se_log_or(40, 60, 55, 45),
                  "fisher_z": ca.fisher_z(0.75),
                  "se_fisher_z": ca.se_fisher_z(103),
                  "r_back": ca.r_from_fisher_z(0.973)}
out["meta_convert"] = {
    "sd_logistic": ca.LOGISTIC_SD,
    "d_logit": ca.d_from_log_or(-0.39, "logit"),
    "d_cox": ca.d_from_log_or(-0.39, "cox"),
    "se_d_logit": ca.se_d_from_se_log_or(0.3, "logit"),
    "se_d_cox": ca.se_d_from_se_log_or(0.3, "cox"),
    "d_probit": ca.d_probit(0.25, 0.33),
    "se_d_probit": ca.se_d_probit(0.25, 0.33, 42, 29),
    "ln_or_logit": ca.log_or_from_d(0.4, "logit"),
    "ln_or_cox": ca.log_or_from_d(0.4, "cox"),
    "se_ln_or_logit": ca.se_log_or_from_se_d(0.15, "logit"),
    "or_from_rr": ca.or_from_rr(0.727, 0.55),
    "rr_from_or": ca.rr_from_or(0.6768, 0.55),
    "r_from_d": ca.r_from_d(0.6, 30, 70),
    "r_from_d_eq": ca.r_from_d(0.6),
    "se_r_from_d": ca.se_r_from_se_d(0.6, 0.2, 30, 70),
    "d_from_r": ca.d_from_r_pointbiserial(0.287),
    "se_d_from_r": ca.se_d_from_se_r(0.3, 0.1)}
ys = [-0.23, 0.25, 0.08, 0.10, 0.20, 0.22]
ses = [0.32, 0.31, 0.11, 0.26, 0.17, 0.24]
ws = [ca.fixed_effect_weight(s) for s in ses]
mm = ca.mean_effect_size(ys, ws)
qq = ca.q_statistic(ys, ws)
tau2 = ca.tau2_dersimonian_laird(ys, ws)
qwb = ca.q_within_between([ys[:3], ys[3:]], [ws[:3], ws[3:]])
out["meta_pool"] = {"mean": mm["mean"], "se": mm["se"], "z": mm["z"],
                    "q": qq["q"], "i2": ca.i_squared(qq["q"], qq["df"]),
                    "tau2": tau2,
                    "w_random_first": ca.random_effects_weight(ses[0], tau2),
                    "q_within": qwb["q_within"], "q_between": qwb["q_between"]}
wmat = [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]]
out["spatial"] = {"i_checker": ca.morans_i([1.0, -1.0, -1.0, 1.0], wmat),
                  "i_data": ca.morans_i([2.0, 5.0, 1.0, 4.0], wmat),
                  "expected": ca.morans_i_expected(6)}
ring = np.zeros((6, 6))
for i in range(6):
    ring[i, (i - 1) % 6] = ring[i, (i + 1) % 6] = 0.5
xb = np.linspace(-1, 1, 6)
e = np.array([0.1, -0.2, 0.05, 0.0, -0.1, 0.15])
out["sar"] = {"y": ca.spatial_lag_reduced_form(0.4, ring, xb, e).tolist()}

json.dump(out, open(sys.argv[1], "w"), indent=1)
print("emitted", sum(len(v) if isinstance(v, dict) else 1 for v in out.values()),
      "values in", len(out), "groups")
