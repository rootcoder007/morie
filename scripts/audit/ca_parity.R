# R-side parity check for the Criminology shelf (vs ca_emit_python_values.py).
# Usage: Rscript ca_parity.R <python_values.json> <path_to_ca_crim_native.R>

args <- commandArgs(trailingOnly = TRUE)
py <- jsonlite::fromJSON(args[[1]])
source(args[[2]])

fails <- 0L
checked <- 0L
cmp <- function(name, r_val, py_val, tol = 1e-12) {
  checked <<- checked + 1L
  r_val <- as.numeric(r_val)
  py_val <- as.numeric(py_val)
  rel <- abs(r_val - py_val) / pmax(abs(py_val), 1e-300)
  if (any(rel > tol & abs(r_val - py_val) > tol)) {
    fails <<- fails + 1L
    cat(sprintf("FAIL %s: R=%.15g PY=%.15g rel=%.3g\n",
                name, r_val[1], py_val[1], max(rel)))
  }
}

x21 <- c(1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5)
y21 <- c(1, 0, 1, 2, 2, 0, 1, 1, 3, 3, 3, 4, 2, 4, 4, 4, 5, 6, 9, 7)

fit <- morie_ols_simple(x21, y21)
for (k in c("b1", "b0", "r", "se_b1", "t"))
  cmp(paste0("ols_", k), fit[[k]], py$ols_simple[[k]])

tw <- morie_ols_two_iv(0.7156, 0.7616, 0.6280, 2.300, 0.9631, 2.658)
cmp("two_iv_b1", tw$b1, py$ols_two_iv$b1)
cmp("two_iv_b2", tw$b2, py$ols_two_iv$b2)

fi <- morie_fit_indices(y21, fit$b0 + fit$b1 * x21, k = 1)
for (k in c("var_total", "var_model", "var_resid", "r2", "adj_r2",
            "f_from_r2"))
  cmp(paste0("fit_", k), fi[[k]], py$fit_indices[[k]])

fc <- morie_f_change(ss_resid_restricted = 80, ss_resid_full = 60,
                     r2_full = 0.4, r2_restricted = 0.3,
                     k_full = 5, k_restricted = 2, n = 100)
cmp("f_ss", fc$f_ss, py$f_change$f_ss)
cmp("f_r2", fc$f_r2, py$f_change$f_r2)

cmp("std_ols", morie_std_coef(2.0, 1.5, 3.0), py$std_coef$ols)
cmp("std_logistic", morie_std_coef(0.5, 0.4), py$std_coef$logistic)
cmp("std_gelman", morie_std_coef(0.5, 0.4, gelman = TRUE),
    py$std_coef$gelman)

vt <- morie_vif_tolerance(0.06)
cmp("tolerance", vt$tolerance, py$vif_tol$tolerance)
cmp("vif", vt$vif, py$vif_tol$vif)

ll <- morie_logit_link(p = 0.3, xb = 0.205, b = 0.805)
cmp("logit", ll$logit, py$logit_link$logit)
cmp("odds", ll$odds, py$logit_link$odds)
cmp("p_from_xb", ll$p_from_xb, py$logit_link$p_from_xb)
cmp("odds_ratio", ll$odds_ratio, py$logit_link$odds_ratio)

le <- morie_logistic_effects(0.5, 0.8, 0.332, 80, 100, 528.171, 492.513,
                             neg2ll_reduced = 499.447, n = 417)
cmp("dm", le$dm, py$logistic_effects$dm)
cmp("pct", le$pct_correct, py$logistic_effects$pct)
cmp("chi2", le$model_chi2, py$logistic_effects$chi2)
cmp("wald", morie_logistic_effects(0.5, 0.805, 0.332, 80, 100, 528.171,
                                   492.513)$wald,
    py$logistic_effects$wald)
cmp("cox_snell", le$cox_snell_r2, py$logistic_effects$cox_snell)
cmp("lr", le$lr_chi2, py$logistic_effects$lr)

ml <- morie_mlogit_probs(c(0.0, 1.2, -0.4))
cmp("mlogit_probs", ml$probs, py$mlogit$probs)
cmp("cond_or", ml$or_matrix[2, 3], py$mlogit$cond_or)

od <- morie_ordinal_logit_ca(c(.15, .30, .35, .20), 2, tau_m = 0.5,
                             xb = 0.6)
cmp("cum_prob", od$cum_prob, py$ordinal$cum_prob)
od1 <- morie_ordinal_logit_ca(c(.15, .30, .35, .20), 1)
cmp("cum_logit", od1$cum_logit, py$ordinal$cum_logit)
cmp("ord_plus", od$logit_plus, py$ordinal$plus)
cmp("ord_minus", od$logit_minus, py$ordinal$minus)

cg <- morie_count_glm(b0 = -1.0, b1 = 0.736, x1 = 2.0, exposure = 100,
                      y = c(2, 0, 3, 1, 4, 2), yhat = c(1.5, 0.8, 2.5, 1.2,
                                                        3.5, 2.0),
                      k = 1, se = 0.083, mu = 2.0, alpha = 0.5)
cmp("cg_predict", cg$predict, py$count_glm$predict)
cmp("cg_irr", cg$irr, py$count_glm$irr)
cmp("cg_offset", cg$predict_offset, py$count_glm$offset)
cmp("cg_theta", cg$theta, py$count_glm$theta)
cmp("cg_se_quasi", cg$se_quasi, py$count_glm$se_quasi)
cmp("cg_negbin", cg$negbin_var, py$count_glm$negbin_var)

hl <- morie_hlm_components(3.9096, 0.27, 117.41, ll_null = -1871.73,
                           ll_full = -1777.35)
cmp("sigma2_u", hl$sigma2_u, py$hlm$sigma2_u)
cmp("icc", morie_hlm_components(0.301, 0.270, 1)$icc, py$hlm$icc, 1e-6)
cmp("lr_chi2", hl$lr_chi2, py$hlm$lr_chi2)

pw <- morie_power_ttest_crim(d = 0.2, n1 = 100, n2 = 100, t_cv = 1.6526,
                             df = 198, f = 0.25, n_total = 300, r = 0.3,
                             n = 100)
cmp("delta_d", pw$delta_d, py$power$delta_d)
cmp("t_beta", pw$t_beta, py$power$t_beta)
cmp("beta", pw$beta, py$power$beta, 1e-8)
cmp("power", pw$power, py$power$power, 1e-8)
cmp("lambda", pw$lambda, py$power[["lambda"]])
cmp("delta_r", pw$delta_r, py$power$delta_r)
cmp("r2_f2", pw$r2_f2, py$power$r2_f2)

rc <- morie_rct_tests(r_yt = -0.25, r_yx = -0.50, r_tx = 0.50, s_y = 1,
                      s_t = 1, m1 = 127.8, m2 = 132.3, s1 = 10.4,
                      s2 = 9.8, n1 = 25, n2 = 30, a = 30, b = 20, c = 15,
                      d = 35, differences = c(1.0, 2.0, 0.5, 1.5, 1.0))
cmp("b_t", rc$b_t, py$rct$b_t)
cmp("b_t_random", morie_rct_tests(r_yt = 0.3, s_y = 2, s_t = 1)$b_t_random,
    py$rct$b_t_random)
cmp("t_ind", rc$t, py$rct$t)
cmp("s_pooled", rc$s_pooled, py$rct$s_pooled)
cmp("chi2_2x2", rc$chi2, py$rct$chi2)
cmp("t_paired", rc$t_paired, py$rct$t_paired)

an <- morie_experiment_anova(groups = list(c(1, 2, 3, 2.5), c(4, 5, 6, 5.5),
                                           c(7, 8, 9, 8.5)))
cmp("ms_between", an$ms_between, py$anova$ms_between)
cmp("ms_within", an$ms_within, py$anova$ms_within)
cmp("anova_f", an$f, py$anova$f)

blk <- morie_experiment_anova(y = c(1.0, 2.1, 6.2, 6.9, 11.1, 12.0, 15.8,
                                    17.1),
                              treatment = rep(c(0, 1), 4),
                              block = rep(0:3, each = 2))
cmp("block_f", blk$f_treatment, py$block$f_treatment)
cmp("block_ss", blk$ss_block, py$block$ss_block)

cmp("psm_bias", morie_psm_balance(0.5, 0.4, 0.2, 0.2), py$psm$bias)

es <- morie_meta_effect_sizes(m1 = 127.8, m2 = 132.3, s1 = 10.4, s2 = 9.8,
                              n1 = 25, n2 = 30, t_value = py$rct$t)
cmp("meta_d", es$d, py$meta_es$d)
cmp("meta_sp", es$s_pooled, py$meta_es$s_pooled)
cmp("meta_j", es$j, py$meta_es$j)
cmp("meta_g", es$g, py$meta_es$g)
cmp("meta_se_g", es$se_g, py$meta_es$se_g)
cmp("meta_d_from_t", es$d_from_t, py$meta_es$d_from_t)
es2 <- morie_meta_effect_sizes(a = 40, b = 60, c = 55, d = 45)
cmp("meta_rr", es2$rr, py$meta_es$rr)
cmp("meta_or", es2$or, py$meta_es[["or"]])
cmp("meta_se_ln_rr", es2$se_ln_rr, py$meta_es$se_ln_rr)
cmp("meta_se_ln_or", es2$se_ln_or, py$meta_es$se_ln_or)
es3 <- morie_meta_effect_sizes(n1 = 103, n2 = 103, r = 0.75)
cmp("fisher_z", es3$fisher_z, py$meta_es$fisher_z)
cmp("se_fisher_z", es3$se_fisher_z, py$meta_es$se_fisher_z)
cmp("r_back", tanh(0.973), py$meta_es$r_back)

cv <- morie_meta_convert(ln_or = -0.39, se_ln_or = 0.3, p1 = 0.25,
                         p2 = 0.33, n1 = 42, n2 = 29, d = 0.4,
                         se_d = 0.15)
cmp("sd_logistic", cv$sd_logistic, py$meta_convert$sd_logistic)
cmp("d_logit", cv$d_logit, py$meta_convert$d_logit)
cmp("d_cox", cv$d_cox, py$meta_convert$d_cox)
cmp("se_d_logit", cv$se_d_logit, py$meta_convert$se_d_logit)
cmp("se_d_cox", cv$se_d_cox, py$meta_convert$se_d_cox)
# probit pair: Python uses the Acklam qnorm approximation (|err| < 1.2e-9)
cmp("d_probit", cv$d_probit, py$meta_convert$d_probit, 1e-8)
cmp("se_d_probit", cv$se_d_probit, py$meta_convert$se_d_probit, 1e-8)
cmp("ln_or_logit", cv$ln_or_logit, py$meta_convert$ln_or_logit)
cmp("ln_or_cox", cv$ln_or_cox, py$meta_convert$ln_or_cox)
cmp("se_ln_or_logit", cv$se_ln_or_logit, py$meta_convert$se_ln_or_logit)
cv2 <- morie_meta_convert(rr = 0.727, or_value = 0.6768, p2 = 0.55)
cmp("or_from_rr", cv2$or_from_rr, py$meta_convert$or_from_rr)
cmp("rr_from_or", cv2$rr_from_or, py$meta_convert$rr_from_or)
cmp("r_from_d", morie_meta_convert(d = 0.6, n1 = 30, n2 = 70)$r_from_d,
    py$meta_convert$r_from_d)
cmp("r_from_d_eq", morie_meta_convert(d = 0.6)$r_from_d,
    py$meta_convert$r_from_d_eq)
cmp("se_r_from_d",
    morie_meta_convert(d = 0.6, se_d = 0.2, n1 = 30, n2 = 70)$se_r_from_d,
    py$meta_convert$se_r_from_d)
cmp("d_from_r", morie_meta_convert(r = 0.287)$d_from_r,
    py$meta_convert$d_from_r)
cmp("se_d_from_r", morie_meta_convert(r = 0.3, se_r = 0.1)$se_d_from_r,
    py$meta_convert$se_d_from_r)

mp <- morie_meta_pool(c(-0.23, 0.25, 0.08, 0.10, 0.20, 0.22),
                      c(0.32, 0.31, 0.11, 0.26, 0.17, 0.24),
                      groups = c(1, 1, 1, 2, 2, 2))
cmp("pool_mean", mp$mean, py$meta_pool$mean)
cmp("pool_se", mp$se, py$meta_pool$se)
cmp("pool_z", mp$z, py$meta_pool$z)
cmp("pool_q", mp$q, py$meta_pool$q)
cmp("pool_i2", mp$i2, py$meta_pool$i2)
cmp("pool_tau2", mp$tau2, py$meta_pool$tau2)
cmp("pool_w_random", mp$weights_random[1], py$meta_pool$w_random_first)
cmp("pool_q_within", mp$q_within, py$meta_pool$q_within)
cmp("pool_q_between", mp$q_between, py$meta_pool$q_between)

wmat <- matrix(c(0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0), 4, 4,
               byrow = TRUE)
mi <- morie_morans_i(c(1, -1, -1, 1), wmat)
cmp("i_checker", mi$i, py$spatial$i_checker)
cmp("i_data", morie_morans_i(c(2, 5, 1, 4), wmat)$i, py$spatial$i_data)
cmp("i_expected", morie_morans_i(rep(c(1, 2), 3), diag(6) * 0 + 1)$expected,
    py$spatial$expected)

ring <- matrix(0, 6, 6)
for (i in 1:6) {
  ring[i, ((i - 2) %% 6) + 1] <- 0.5
  ring[i, (i %% 6) + 1] <- 0.5
}
xb <- seq(-1, 1, length.out = 6)
e <- c(0.1, -0.2, 0.05, 0.0, -0.1, 0.15)
cmp("sar_y", morie_sar_lag(0.4, ring, xb, e), py$sar$y)

cat(sprintf("PARITY: %d comparisons, %d failed\n", checked, fails))
if (fails > 0) quit(status = 1)
