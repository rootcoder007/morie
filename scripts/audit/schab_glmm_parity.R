# Three-way parity: the R arm against values computed by the Python arm.
source("/home/rootcoder/work/morie/r-package/morie/R/aaa_schab_glmm_shared.R")

py <- jsonlite::fromJSON("/tmp/py_glmm_values.json")
FAIL <- character(0)

chk <- function(name, got, want, tol = 1e-12) {
  got <- as.numeric(got); want <- as.numeric(want)
  if (length(got) != length(want)) {
    cat(sprintf("  %-54s FAIL (length %d vs %d)\n", name, length(got),
                length(want)))
    FAIL <<- c(FAIL, name); return(invisible())
  }
  d <- max(abs(got - want))
  rel <- max(abs(got - want) / pmax(abs(want), 1e-300))
  ok <- d <= tol || rel <= tol
  cat(sprintf("  %-54s %s  max|d|=%.3e\n", name, if (ok) "PASS" else "FAIL", d))
  if (!ok) FAIL <<- c(FAIL, name)
}

n <- py$n; nb <- py$nb
X <- matrix(py$X, ncol = 2); S <- py$S; z <- py$z
Sigma_S <- matrix(py$Sigma_S, nrow = n)
beta <- py$beta
mu <- exp(as.numeric(X %*% beta) + S)

cat("\n[6.3.4] conditional specification\n")
chk("eq (6.73) conditional mean",
    .schab_conditional_mean(X, beta, S, "log"), py$cond_mean)
chk("eq (6.74) conditional variance",
    .schab_conditional_variance(mu, 2, "poisson"), py$cond_var)
chk("naive g^-1(x'beta)", .schab_naive_marginal_mean(X, beta, "log"), py$naive)
mom <- .schab_marginal_moments_lognormal(X, beta, 0.4, 1,
                                         rho = exp(-abs(outer(seq(0, 8, length.out = n),
                                                              seq(0, 8, length.out = n), "-")) / 2.5))
chk("Example 6.6 marginal mean", mom$mean, py$marg_mean)
chk("Example 6.6 marginal variance (squared term)", mom$variance, py$marg_var)
chk("Example 6.6 marginal covariance", as.numeric(mom$covariance), py$marg_cov)

cat("\n[6.3.5] pseudo-likelihood\n")
chk("eq (6.78) pseudo-data", .schab_pseudo_data(z, mu, "log"), py$pseudo)
chk("eq (6.79) Sigma_mu (pseudo scale)",
    as.numeric(.schab_sigma_mu(mu, 1, "poisson", "log")), py$sigma_mu)
chk("data-scale covariance (what the scores need)",
    as.numeric(.schab_data_covariance(mu, 1, "poisson")), py$data_cov)
fit <- .schab_fit_pseudo_likelihood(z, X, Sigma_S, family = "poisson")
chk("eq (6.80) beta", fit$beta, py$fit_beta, 1e-9)
chk("eq (6.81) S_hat", fit$S, py$fit_S, 1e-9)
chk("fitted mu", fit$mu, py$fit_mu, 1e-9)
chk("standard errors", fit$se_beta, py$fit_se, 1e-9)
chk("iterations agree", fit$n_iter, py$fit_iter, 0)
chk("eq (6.84) REML objective",
    .schab_reml_objective(X, fit$Sigma_nu, fit$pseudo_data), py$reml, 1e-9)
sc <- .schab_pql_score(z, X, fit$beta, fit$S, Sigma_S, "poisson", "log")
chk("PQL score for beta", sc$score_beta, py$score_beta, 1e-7)
chk("PQL score for S", sc$score_S, py$score_S, 1e-7)

cat("\n[6.3.6] prediction\n")
pr <- .schab_predict_glm(1.15, 0.08, 2.5, "log")
chk("eqs (6.90)/(6.91)/(6.87)",
    c(pr$prediction, pr$mspe, pr$inverse_link_prediction), py$pred)

cat("\n[CAR family]\n")
A <- matrix(py$A, nrow = nb); u <- py$u; v <- py$v
R <- .schab_neighbour_structure(A)
chk("structure matrix R", as.numeric(R), py$R)
chk("ICAR covariance (Moore-Penrose)",
    as.numeric(.schab_icar_covariance(R)), py$icar_cov, 1e-10)
fc <- .schab_icar_full_conditional(u, A)
chk("ICAR conditional mean", fc$mean, py$icar_fc_mean)
chk("ICAR conditional variance", fc$variance, py$icar_fc_var)
chk("LCAR precision", as.numeric(.schab_lcar_precision(R, 0.6)), py$lcar_Q)
lfc <- .schab_lcar_full_conditional(u, A, 0.6)
chk("LCAR conditional mean", lfc$mean, py$lcar_fc_mean)
chk("LCAR conditional variance", lfc$variance, py$lcar_fc_var)
chk("SMR", .schab_smr(py$Y, py$E), py$smr)

cat("\n[BYM 1991]\n")
chk("eq (4.2) ICAR log prior",
    .schab_bym_icar_log_prior(u, A, 0.129), py$bym_icar_logprior)
chk("eq (4.4) median log prior",
    .schab_bym_median_log_prior(u, A, 0.129), py$bym_median_logprior)
bym <- .schab_bym_map(py$Y, py$E, A, 0.129, 0.011)
chk("BYM MAP u*", bym$u, py$bym_u, 1e-9)
chk("BYM MAP v*", bym$v, py$bym_v, 1e-9)
chk("BYM MAP x* = u* + v*", bym$x, py$bym_x, 1e-9)
chk("eq (4.5) log posterior at the MAP", bym$log_posterior, py$bym_logpost, 1e-9)
cat(sprintf("  %-54s sum_v=%.3e  fitted=%.9f observed=%.9f\n",
            "BYM identities hold in the R arm too", bym$sum_v,
            bym$fitted_total, bym$observed_total))
if (abs(bym$sum_v) > 1e-8 || abs(bym$fitted_total - bym$observed_total) > 1e-7) {
  FAIL <- c(FAIL, "BYM stationarity identities")
}

cat("\n[temporal + space-time]\n")
Rt1 <- .schab_random_walk_structure(6, 1)
Rt2 <- .schab_random_walk_structure(6, 2)
chk("RW1 structure", as.numeric(Rt1), py$rw1)
chk("RW2 structure", as.numeric(Rt2), py$rw2)
i4 <- .schab_interaction_structure(R, Rt1, "IV")
chk("Type IV rank", i4$rank, py$i4_rank, 0)
chk("Type IV rank deficiency", i4$rank_deficiency, py$i4_def, 0)
c4 <- .schab_null_space_constraints(i4$structure)
chk("Type IV constraint count", c4$n_constraints, py$c4_n, 0)
chk("eq (12) projection", .schab_apply_sum_to_zero(seq(0, nb * 6 - 1), c4$A),
    py$sum_to_zero, 1e-9)
chk("eq (9) linear trend",
    as.numeric(.schab_linear_trend_log_risk(0.1, u, 0.05, v, 0:5)), py$trend)
chk("eq (10) nonparametric",
    as.numeric(.schab_nonparametric_log_risk(0.1, u, seq(0, 0.5, length.out = 6),
                                             0:5)), py$nonpar)

cat(sprintf("\n%s  %d failed\n", strrep("=", 62), length(FAIL)))
if (length(FAIL)) {
  cat("FAILED:", paste(FAIL, collapse = ", "), "\n")
  quit(status = 1)
}
