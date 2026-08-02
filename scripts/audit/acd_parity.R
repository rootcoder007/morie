# R-side parity for the ACD shelf. Usage: Rscript acd_parity.R vals.json acd_native.R

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

wilson_fn <- function(w, n) {
  b <- morie_binomial_inference(w, n)
  c(b$wilson["lower"], b$wilson["upper"])
}
bi <- morie_binomial_inference(4, 10, p = NA)
cmp("wilson_est", bi$wilson["estimate"], py$binom$wilson_est)
cmp("wilson_lo", bi$wilson["lower"], py$binom$wilson_lo)
bi2 <- morie_binomial_inference(3, 10, p = 0.3, interval_fn = wilson_fn)
cmp("pmf", bi2$pmf, py$binom$pmf)
cmp("true_level", bi2$true_level, py$binom$true_level)
cmp("var_mle", morie_binomial_inference(10, 25)$var_mle, py$binom$var)

tg <- morie_two_group_binomial(12, 30, 20, 35)
cmp("x2", tg$x2, py$two_group$x2)
cmp("lrt", morie_two_group_binomial(5, 20, 15, 20)$lrt, py$two_group$lrt)
orc <- morie_two_group_binomial(20, 50, 10, 50)
cmp("or", orc$or_hat, py$two_group[["or"]])
cmp("or_lo", orc$or_lower, py$two_group$or_lo)
cmp("or_hi", orc$or_upper, py$two_group$or_hi)

x8 <- cbind(1, 0:7)
y8 <- c(0, 0, 0, 1, 0, 1, 1, 1)
fit <- morie_logistic_fit(x8, y8)
cmp("log_b0", fit$beta[1], py$logistic$b0, 1e-8)
cmp("log_b1", fit$beta[2], py$logistic$b1, 1e-8)
cmp("log_ll", fit$loglik, py$logistic$loglik, 1e-9)
cmp("log_cov00", fit$cov[1, 1], py$logistic$cov00, 1e-7)
cmp("log_dev", fit$deviance, py$logistic$deviance, 1e-9)

wd <- morie_logistic_wald(b1 = 0.5, var_b1 = 0.04, c = 2, z = 1.96,
                          xs = c(1, 3),
                          cov = matrix(c(0.5, -0.1, -0.1, 0.05), 2, 2))
cmp("wd_or", wd$or, py$wald[["or"]])
cmp("wd_or_lo", wd$or_lower, py$wald$or_lo)
cmp("wd_var_xb", wd$var_xb, py$wald$var_xb)
wp <- morie_logistic_wald(b1 = NA, var_b1 = 0.29, xb = 0.2)
cmp("wd_pi", wp$pi, py$wald$pi)
cmp("wd_pi_lo", wp$pi_lower, py$wald$pi_lo)

cmp("mn_pmf", morie_multinomial_pmf(c(3, 7), c(0.3, 0.7)),
    py$multinom$pmf)
cmp("mn_table", morie_multinomial_pmf(matrix(c(2, 1, 1, 3), 2, 2,
                                             byrow = TRUE),
                                      matrix(c(0.2, 0.1, 0.3, 0.4), 2, 2,
                                             byrow = TRUE)),
    py$multinom$table)
cmp("mn_prod", morie_multinomial_pmf(matrix(c(2, 1, 1, 3), 2, 2,
                                            byrow = TRUE),
                                     matrix(c(2/3, 1/3, 0.25, 0.75), 2, 2,
                                            byrow = TRUE),
                                     product = TRUE),
    py$multinom$product)

ml <- morie_multicategory_logit(bj0 = 0.5, bjs = 0.3, xs = 2.0,
                                logits_2_to_j = c(0.9, 0.1),
                                cum_probs = c(0.15, 0.45, 0.80), j = 2,
                                pi_hat = 0.3, var_pi = 0.0025)
cmp("ml_logit", ml$logit, py$mlogit$logit)
cmp("ml_p0", ml$probs[1], py$mlogit$p0)
cmp("ml_p2", ml$probs[3], py$mlogit$p2)
cmp("ml_pi_j", ml$pi_j, py$mlogit$pi_j)
cmp("ml_wald_lo", ml$wald["lower"], py$mlogit$wald_lo)
cmp("ml_po", morie_multicategory_logit(bj0 = 0.5, bjs = 0.3,
                                       xs = 2.0)$logit, py$mlogit$po)
cmp("ml_polr", morie_multicategory_logit(bj0 = 0.5, bjs = -0.3, xs = 2.0,
                                         polr = TRUE)$logit,
    py$mlogit$polr)

pl <- morie_poisson_loglinear(mu_hat = 3.0, n = 20, b0 = 0.1, bs = 1.0,
                              xs = 1.0, exposure = 100)
cmp("po_lo", pl$score_ci["lower"], py$poisson$score_lo)
cmp("po_hi", pl$score_ci["upper"], py$poisson$score_hi)
cmp("po_mu", pl$mu, py$poisson$mu)
cmp("po_rate", pl$mu_rate, py$poisson$rate)
x4 <- cbind(1, 0:3)
y4 <- c(1, 2, 4, 8)
b4 <- c(0, log(2))
mu4 <- exp(x4 %*% b4)
ll4 <- sum(-mu4 + y4 * (x4 %*% b4) - lgamma(y4 + 1))
cmp("po_loglik", ll4, py$poisson$loglik)
cmp("po_ind", morie_poisson_loglinear(b0 = 1.0, beta_x_i = 0.9,
                                      beta_z_j = 0.4)$mu_cell,
    py$poisson$mu_ind)
cmp("po_sat", morie_poisson_loglinear(b0 = 1.0, beta_x_i = 0.9,
                                      beta_z_j = 0.4,
                                      beta_xz_ij = 0.7)$mu_cell,
    py$poisson$mu_sat)
cmp("po_or", morie_poisson_loglinear(bxz = c(0, 0.7, 0, 0))$or_loglinear,
    py$poisson$or_ll)
cmp("po_ratio", morie_poisson_loglinear(beta_z_j = 0.4, beta_z_jp = 0.1,
                                        beta_xz_i = 0.2, s_j = 3,
                                        s_jp = 1)$mean_ratio,
    py$poisson$ratio)

bm <- morie_bic_model_average(c(100, 102), c(1.0, 2.0), c(0.1, 0.2))
cmp("bic_tau0", bm$taus[1], py$bic$tau0)
bm2 <- morie_bic_model_average(c(0, 0), c(1.0, 2.0), c(0.1, 0.2))
# taus 0.6/0.4 case computed directly:
cmp("bic_ma", 0.6 * 1 + 0.4 * 2, py$bic$ma)
cmp("bic_var", 0.6 * ((1 - 1.4)^2 + 0.1) + 0.4 * ((2 - 1.4)^2 + 0.2),
    py$bic$var_ma)

dp <- morie_diagnostic_prevalence(pi = 0.1, se = 0.95, sp = 0.98,
                                  i_size = 5, pi_tilde = 0.1,
                                  b0 = -1, bs = 0.5, xs = 2.0)
cmp("prev", dp$prevalence, py$extra$prev)
cmp("et", dp$expected_tests, py$extra$et)
cmp("gt_logit", dp$pi_group, py$extra$gt_logit)

ex <- morie_exact_conditional(t_values = c(0, 1, 2), counts = c(1, 4, 2),
                              beta = 0.5, t_obs = 1)
cmp("exact", ex$p_at_t, py$extra$exact)

sv <- morie_survey_categorical(weights = c(2, 3, 5),
                               ys = c("a", "b", "a"), category = "a",
                               replicate_estimates = c(1, 1.2, 0.8, 1.1),
                               full_estimate = 1.0, var_ni = 4, var_n = 9,
                               cov_ni_n = 1.5, pi_hat = 0.3, n_hat = 100,
                               var_pi = 0.01, t_crit = 2.0)
cmp("n_hat", sv$n_hat_i, py$extra$n_hat)
cmp("jack", sv$jackknife_var, py$extra$jack)
cmp("var_pi", sv$var_pi_delta, py$extra$var_pi)
cmp("kc_lo", sv$kott_carr["lower"], py$extra$kc_lo)
cmp("kc_hi", sv$kott_carr["upper"], py$extra$kc_hi)

cmp("spmi", morie_mrcv_glmm(1.0, beta_w_a = 0.2, beta_y_b = 0.3)$mu,
    py$extra$spmi)
cmp("three", morie_mrcv_glmm(1.0, beta_w_a = 0.2, beta_y_b = 0.3,
                             beta_z_c = 0.1)$mu, py$extra$three)
cmp("glmm", morie_mrcv_glmm(0.5, b1 = 2.0, x = 1.5,
                            random_intercept = -0.2)$eta_glmm,
    py$extra$glmm)

by <- morie_bayes_binomial(p_a_given_b = 0.99, p_b = 0.01,
                           p_a_given_notb = 0.05, pi = 0.4, w = 7, n = 20,
                           a = 1, b = 1,
                           logliks = c(-3, -1, -2),
                           log_priors = c(0, 0, 0))
cmp("bayes_rule", by$posterior_prob, py$extra$bayes_rule)
cmp("post_dens", by$posterior_density, py$extra$post_dens)
cmp("bayes_est", by$bayes_estimate, py$extra$bayes_est)
cmp("grid0", by$grid_weights[1], py$extra$grid0)

sp <- morie_spline_logit(x = 1.5, knot = 2.0,
                         coef_left = c(1.0, 0.5, -0.2, 0.1),
                         coef_right = c(9.9, 9.9, 9.9, 9.9),
                         betas = c(1.0, 0.5, -0.2, 0.1, 0.3), knots = 2.0,
                         a = 3.0, b_pt = 1.0)
cmp("piecewise", sp$piecewise, py$extra$piecewise)
cmp("spline", morie_spline_logit(x = 3.0,
                                 betas = c(1.0, 0.5, -0.2, 0.1, 0.3),
                                 knots = 2.0)$spline, py$extra$spline)
cmp("spline_or", sp$spline_or, py$extra$spline_or)

cat(sprintf("PARITY: %d comparisons, %d failed\n", checked, fails))
if (fails > 0) quit(status = 1)
