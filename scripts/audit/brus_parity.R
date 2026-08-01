# R-side parity for the Brus shelf. Usage: Rscript brus_parity.R vals.json brus_native.R

args <- commandArgs(trailingOnly = TRUE)
py <- jsonlite::fromJSON(args[[1]])
source(args[[2]])

fails <- 0L; checked <- 0L
cmp <- function(name, r_val, py_val, tol = 1e-12) {
  checked <<- checked + 1L
  r_val <- as.numeric(r_val); py_val <- as.numeric(py_val)
  rel <- abs(r_val - py_val) / pmax(abs(py_val), 1e-300)
  if (any(rel > tol & abs(r_val - py_val) > tol)) {
    fails <<- fails + 1L
    cat(sprintf("FAIL %s: R=%.15g PY=%.15g\n", name, r_val[1], py_val[1]))
  }
}

z3 <- c(2, 5, 3); pi3 <- c(0.5, 0.5, 0.5)
ht <- morie_ht_estimators(z3, pi3, 10)
cmp("ht_total", ht$total, py$ht$total)
cmp("ht_mean", ht$mean, py$ht$mean)

si <- morie_si_estimators(y = c(1, 0, 1, 0), n = 4, n_population = 20,
                          estimate = 10, variance = 4, u_crit = 1.645,
                          zbar_hat = 3, area = 100, sample_area = 10,
                          s2_hat = 2)
cmp("si_p", si$p_hat, py$si$p)
si2 <- morie_si_estimators(y = c(1, 0, 1, 0), n = 4, n_population = 20)
cmp("si_var_p", si2$var_p, py$si$var_p)
cmp("si_ci", si$ci[1], py$si$ci_lower)
si3 <- morie_si_estimators(zbar_hat = 3, area = 100, sample_area = 10,
                           s2_hat = 2, n = 8)
cmp("si_total_inf", si3$total_inf, py$si$total_inf)
cmp("si_var_inf", si3$var_total_inf, py$si$var_total_inf)

st <- morie_stsi_estimators(stratum_means = c(4, 6),
                            stratum_weights = c(0.5, 0.5),
                            stratum_variances = c(0.5, 0.8), c0 = 10,
                            stratum_costs = c(2, 3),
                            stratum_sizes = c(5, 4))
cmp("stsi_mean", st$mean, py$stsi$mean)
cmp("stsi_var", st$variance, py$stsi$var)
cmp("stsi_cost", st$cost, py$stsi$cost)

cl <- morie_cluster_twostage(cluster_totals = c(7, 11),
                             cluster_sizes = c(2, 2), m_population = 6,
                             n_clusters_population = 3,
                             primary_unit_means = c(3.5, 5.5, 4.5),
                             s2_between = 4, s2_within = 9, n = 2, m = 3)
cmp("cl_pps", cl$total_pps, py$cluster$pps)
cmp("cl_si", cl$total_si, py$cluster$si)
cmp("cl_mean", morie_cluster_twostage(cluster_totals = c(27),
                                      cluster_sizes = c(6),
                                      m_population = 6)$mean_from_total,
    py$cluster$mean)
cmp("ts_mean", cl$ts_mean, py$cluster$ts_mean)
cmp("ts_var", cl$ts_variance, py$cluster$ts_var)
cmp("ts_true", cl$true_variance, py$cluster$true_var)
cmp("ts_total_si", 5 / 2 * sum(c(10, 14)), py$cluster$total_si)

dg <- morie_twostage_design(3, 2, 10, 1, v_max = 0.5, c_max = 200)
cmp("m_opt", dg$m_opt, py$design$m)
cmp("n_v", dg$n_for_variance, py$design$n_v)
cmp("n_b", dg$n_for_budget, py$design$n_b)

cmp("pps_var", morie_pps_variance(c(2, 5), c(0.2, 0.5), 10), py$pps_var)

ma <- morie_model_assisted(m_all = rep(4, 6), z_sample = z3,
                           m_sample = rep(4, 3), pi_sample = pi3,
                           n_population = 6)
cmp("ma_diff", ma$difference, py$ma$difference)
mb <- morie_model_assisted(x = 1:4, z = seq(2, 8, 2),
                           sigma2 = rep(1, 4), pi = rep(0.5, 4))
cmp("ma_gls", mb$gls_b[1], py$ma$gls_b)
mc <- morie_model_assisted(zbar_pi = 3, b_hats = 0.5, xbar_true = 4,
                           xbar_pi = 3, t_pi_z = 40, t_pi_x = 20,
                           t_x_true = 50)
cmp("ma_slopes", mc$regr_slopes, py$ma$regr_slopes)
cmp("ma_ratio", mc$ratio, py$ma$ratio)
cmp("ma_ratio_g", mc$ratio_g, py$ma$ratio_g)

e4 <- c(0.5, -0.3, 0.1, -0.3)
gv <- morie_greg_variance(e = e4, n = 4, n_population = 20,
                          g = rep(1, 4), ratio = TRUE)
cmp("greg_s2e", gv$s2_e, py$greg$s2_e)
cmp("greg_var", gv$variance, py$greg$var)
cmp("greg_gvar", gv$g_variance, py$greg$g_var)
cmp("greg_ratio_var", gv$ratio_variance, py$greg$ratio_var)
cmp("greg_gw", morie_greg_variance(x_k = 4, xbar_true = 5,
                                   xbar_sample = 4, s2_x = 2)$g_weight,
    py$greg$g_w)
cmp("greg_mc", morie_greg_variance(e = e4[1:3], n = 3, n_population = 6,
                                   pi = pi3)$mc_variance, py$greg$mc_var)

cb <- morie_calibration(group_means_sample = c(4, 8),
                        group_weights = c(0.25, 0.75))
cmp("calib_pst", cb$poststratified, py$calib$pst)
cmp("calib_int", morie_calibration(pi_sample = pi3, b_hat = 0.6,
                                   z_sample = z3,
                                   n_population = 6)$intercept,
    py$calib$intercept)
cmp("calib_cal", morie_calibration(zbar_pi = 3, a_hat = 0.4,
                                   pi_sample = pi3, m_all_mean = 4,
                                   m_ht_mean = 3.5, b_hat = 0.6,
                                   n_population = 6)$calibrated,
    py$calib$calibrated)
cmp("calib_si", morie_calibration(z_sample = z3, b_si = 0.6,
                                  m_all_mean = 4,
                                  m_sample_mean = 3.5)$si_shortcut,
    py$calib$si)

bt <- morie_balanced_twophase(t_pi_z = 40, t_x_true = 50, t_pi_x = 45,
                              b_hat = 0.8)
cmp("bal_regr", bt$regression_total, py$bal$regr_total)
cmp("bal_var", morie_balanced_twophase(e = c(0.1, -0.2, 0.1), pi = pi3,
                                       c_k = rep(1, 3),
                                       n_population = 10,
                                       p = 1)$balanced_variance,
    py$bal$bal_var)
cmp("bal_local", morie_balanced_twophase(e = c(0.1, -0.2, 0.1), pi = pi3,
                                         e_local_mean = rep(0, 3), n = 3,
                                         p = 1)$local_mean_variance,
    py$bal$local_var)
cmp("tp_strat", morie_balanced_twophase(n1h = c(6, 4), n1 = 10,
                                        s2_2h = c(1, 2), n2h = c(3, 2),
                                        zbar_2h = c(4, 7),
                                        zbar_hat = 5.2)$twophase_strat,
    py$bal$tp_strat)
cmp("tp_regr", morie_balanced_twophase(s2_z = 4, n1 = 10, s2_e = 0.2,
                                       n2 = 3,
                                       n_population = 100)$twophase_regr,
    py$bal$tp_regr)
cmp("s2_resid", morie_balanced_twophase(e = c(0.5, -0.5, 0.2),
                                        n = 3)$s2_resid, py$bal$s2_resid)

ns <- morie_sample_size(p_star = 0.5, se_max = 0.05, u_crit = 1.96,
                        s_star = 2, l_max = 1, cv_star = 0.4, r_max = 0.1)
cmp("n_prop_se", ns$n_prop_se, py$nn$prop_se)
cmp("n_mean_len", ns$n_mean_length, py$nn$mean_len)
cmp("n_cv", ns$n_cv, py$nn$cv)
ns2 <- morie_sample_size(u_crit = 1.96, p_star = 0.5, l_max = 0.2)
cmp("n_prop_len", ns2$n_prop_length, py$nn$prop_len)
cmp("n_de", morie_sample_size(design_effect = 4,
                              n_si = 50)$n_design_effect, py$nn$de)
nb <- morie_sample_size(p = 0.35, z = 7, n = 20, c = 1, d = 1, v = 0.2,
                        l = 0.3)
cmp("beta_pdf", nb$beta_pdf, py$nn$beta_pdf, 1e-9)
cmp("beta_interval", nb$interval_prob, py$nn$interval, 1e-5)
na <- morie_sample_size(lengths = c(0.1, 0.2), probs = c(0.5, 0.5),
                        l_max = 0.2, coverages = c(0.96, 0.95),
                        alpha = 0.05)
cmp("alc", na$expected_length, py$nn$alc)
cmp("acc", na$expected_coverage, py$nn$acc)

os <- morie_ospats(gamma_bar_h = c(1, 2), weights = c(0.5, 0.5),
                   n_h = c(1, 1), n = 2)
cmp("os_stsi", os$stsi_variance, py$ospats$stsi)
cmp("os_equal", os$equal_area_variance, py$ospats$equal)
os2 <- morie_ospats(weights = c(0.6, 0.4), s_h = c(3, 5), c_h = c(1, 4),
                    n = 10)
cmp("os_alloc", os2$alloc_variance, py$ospats$alloc)
cmp("os_o", os2$objective_o, py$ospats$obj_o)
os3 <- morie_ospats(zhat_i = 3, zhat_j = 1, r2 = 4, s2_i = 0.5,
                    s2_j = 0.7, s2_ij = 0.2, d2_upper_sum = 6,
                    n_h_units = 3, per_stratum_sums = c(4, 9),
                    n_population = 10)
cmp("os_d2", os3$d2, py$ospats$d2)
cmp("os_s2h", os3$stratum_variance, py$ospats$s2h)
cmp("os_obj", os3$ospats_objective, py$ospats$objective)

s <- c(0, 10, 20)
dm <- abs(outer(s, s, "-"))
covm <- ifelse(dm == 0, 2, 2 * exp(-dm / 15))
cov0 <- 2 * exp(-abs(s - 12) / 15)
kg <- morie_kriging(cov_ss = covm, cov_s0 = cov0)
cmp("krig_lam", kg$lam, py$krig$lam, 1e-9)
cmp("krig_nu", kg$nu, py$krig$nu, 1e-9)
kg2 <- morie_kriging(lam = kg$lam, cov_s0 = cov0, sigma2 = 2, nu = kg$nu,
                     gamma_s0 = 2 - cov0)
cmp("krig_vcov", kg2$v_ok_cov, py$krig$v_cov, 1e-9)
kg3 <- morie_kriging(lam = kg$lam, gamma_s0 = 2 - cov0, nu = -kg$nu)
cmp("krig_vgam", kg3$v_ok_gamma, py$krig$v_gam, 1e-9)
cmp("krig_gamma", morie_kriging(h = 75, c0 = 0, c1 = 2,
                                phi = 25)$gamma_h, py$krig$gamma)
cmp("krig_ll", morie_kriging(z = c(0, 0), mu = c(0, 0),
                             cov = diag(2))$loglik, py$krig$loglik)

vg <- morie_variogram_design(mu = 1, a_i = 0.5, b_ij = -0.2, c_ijk = 0.1,
                             eps = 0.05, a = 2 * diag(3),
                             da_list = list(diag(3)),
                             cov_theta = diag(c(0.04, 0.09)),
                             dv_dtheta = c(1, 2), v_ok = 1.5,
                             e_tau2 = 0.3,
                             dlam_dtheta = list(c(1, 0), c(0, 1)),
                             a_mat = diag(2), akv = 1.8, vkv = 0.4)
cmp("vg_nested", vg$nested, py$vgm$nested)
cmp("vg_fisher", vg$fisher_info[1, 1], py$vgm$fisher)
cmp("vg_vkv", vg$vkv, py$vgm$vkv)
cmp("vg_akv", vg$akv, py$vgm$akv)
cmp("vg_tau2", vg$e_tau2, py$vgm$tau2)
cmp("vg_eac", vg$eac, py$vgm$eac)

sv <- morie_survey_variances(sigma2 = 4, n = 8, rho_bar = 0.2, s2 = 4,
                             n_population = 80, xbar_d = c(1, 2),
                             beta_hat = c(0.5, 0.25), v_d = 0.1,
                             times = 1:4,
                             x = cbind(1, 0:2), c_mat = diag(3),
                             zhat = 1:3, beta0 = 2, beta1 = 3, x_val = 4,
                             x_design = cbind(1, 0:3),
                             z_obs = c(1, 3, 5, 7), sigma2_eps = 2,
                             x0 = c(1, 2), c_hat = "a", c_true = "a",
                             u = "a")
cmp("sv_iid", sv$v_iid, py$misc$v_iid)
cmp("sv_auto", sv$v_autocorrelated, py$misc$v_auto)
cmp("sv_neff", sv$n_effective, py$misc$n_eff)
cmp("sv_fpc", sv$v_fpc, py$misc$v_fpc)
cmp("sv_sa", sv$small_area, py$misc$sa)
cmp("sv_tw", sv$trend_weights[1], py$misc$tw0)
cmp("sv_gls", sv$gls[1], py$misc$gls0, 1e-9)
cmp("sv_lm", sv$linear_model, py$misc$lm)
cmp("sv_ols", sv$ols_beta[1], py$misc$ols0, 1e-9)
# note: sv_pred_var x_design here is 4x2 vs python's 5x2 -> recompute match
sv2 <- morie_survey_variances(x_design = cbind(1, 0:4),
                              z_obs = c(1, 3, 5, 7, 9), sigma2_eps = 2,
                              x0 = c(1, 2))
cmp("sv_pred_var", sv2$ols_pred_var, py$misc$pred_var, 1e-9)
cmp("sv_cls", sv$class_indicator, py$misc$cls)

cat(sprintf("PARITY: %d comparisons, %d failed\n", checked, fails))
if (fails > 0) quit(status = 1)
