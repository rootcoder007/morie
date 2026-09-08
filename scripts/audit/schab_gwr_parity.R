# Three-way parity: the R arm against values computed by the Python arm.
source("/home/rootcoder/work/morie/r-package/morie/R/aaa_schab_gwr_shared.R")

py <- jsonlite::fromJSON("/tmp/py_gwr_values.json")
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

n <- py$n
coords <- matrix(py$coords, nrow = n)
X <- matrix(py$X, nrow = n)
y <- py$y
BW <- py$bw
D <- .schab_pairwise_distances(coords)
d_vec <- py$d_vec

cat("\n[distances]\n")
chk("Euclidean distance matrix", as.numeric(D), py$D, 1e-10)

cat("\n[kernels: Charlton white paper / spgwr / GWmodel]\n")
chk("gaussian exp(-0.5 (d/h)^2)",
    .schab_kernel_weights(d_vec, BW, "gaussian"), py$w_gaussian)
chk("Sec. 5.3.2 density form",
    .schab_kernel_weights(d_vec, BW, "gaussian", normalized = TRUE),
    py$w_gaussian_norm)
chk("bisquare (1-(d/h)^2)^2, truncated",
    .schab_kernel_weights(d_vec, BW, "bisquare"), py$w_bisquare)
chk("tricube (1-(d/h)^3)^3, truncated",
    .schab_kernel_weights(d_vec, BW, "tricube"), py$w_tricube)
chk("boxcar indicator", .schab_kernel_weights(d_vec, BW, "boxcar"), py$w_boxcar)
chk("adaptive bandwidth at k = 3, 8, 15",
    vapply(c(3, 8, 15), function(k) .schab_adaptive_bandwidth(D[1, ], k),
           numeric(1)),
    py$adaptive_bw, 1e-10)

cat("\n[Sec. 6.1.3.1 p. 317: the hat matrix and the fit]\n")
fit <- .schab_gwr_fit(y, X, D, BW)
chk("local coefficients beta(s_i)", as.numeric(fit$params), py$params, 1e-9)
chk("hat matrix S", as.numeric(fit$S), py$S, 1e-9)
chk("fitted values S y", fit$fitted, py$fitted, 1e-9)
chk("residuals", fit$resid, py$resid, 1e-9)
chk("tr(S)", fit$tr_S, py$tr_S, 1e-9)
chk("tr(S'S)", fit$tr_STS, py$tr_STS, 1e-9)
chk("effective parameters 2tr(S)-tr(S'S)", fit$effective_parameters, py$enp, 1e-9)
chk("RSS = y'(I-S)'(I-S)y", fit$rss, py$rss, 1e-9)
chk("ML sigma^2 (what the AICc uses)", fit$sigma2, py$sigma2, 1e-10)
chk("Cressie p. 317 sigma^2", fit$sigma2_cressie, py$sigma2_cressie, 1e-10)
chk("rank-deficient local fit count", fit$n_rank_deficient,
    py$n_rank_deficient, 0)
chk("eq (2.16) sigma^2 (n - 2v1 + v2)", fit$sigma2_gwr, py$sigma2_gwr, 1e-10)
chk("effective residual df n - 2v1 + v2", fit$edf_resid, py$edf_resid, 1e-9)
chk("eq (2.15) local standard errors", as.numeric(fit$se_params),
    py$se_params, 1e-9)

cat("\n[criteria: Fotheringham et al. (2002) eqs (2.33), (4.22)]\n")
chk("AICc", .schab_aicc_from_parts(n, fit$sigma2, fit$tr_S), py$aicc, 1e-9)
chk("AIC", .schab_aic_from_parts(n, fit$sigma2, fit$tr_S), py$aic, 1e-9)
chk("leave-one-out CV (gaussian)", .schab_cv_score(y, X, D, BW), py$cv, 1e-9)
chk("leave-one-out CV (bisquare)",
    .schab_cv_score(y, X, D, BW, "bisquare"), py$cv_bisquare, 1e-9)

# The published spgwr NY8 output, recomputed in the R arm as well.
ny <- py$ny8
sigma2_ny <- ny[2] / ny[1]
base <- 2 * ny[1] * log(sqrt(sigma2_ny)) + ny[1] * log(2 * pi) + ny[1]
tr_S_ny <- ny[4] - base
chk("eq (4.22) reproduces spgwr's published AIC 561.6",
    .schab_aic_from_parts(ny[1], sigma2_ny, tr_S_ny), ny[4], 1e-9)
cat(sprintf("  %-54s AICc=%.2f vs published %.0f (tr(S)=%.3f)\n",
            "eq (2.33) reproduces spgwr's published AICc 568",
            .schab_aicc_from_parts(ny[1], sigma2_ny, tr_S_ny), ny[3], tr_S_ny))
if (abs(.schab_aicc_from_parts(ny[1], sigma2_ny, tr_S_ny) - ny[3]) > 0.5) {
  FAIL <- c(FAIL, "published AICc")
}

cat("\n[bandwidth selection: spgwr::gwr.sel interval, golden section]\n")
sel_cv <- .schab_select_bandwidth(y, X, coords, criterion = "cv")
sel_aicc <- .schab_select_bandwidth(y, X, coords, criterion = "aicc")
chk("search interval", sel_cv$bounds, py$sel_bounds, 1e-10)
chk("CV-optimal bandwidth", sel_cv$bandwidth, py$sel_cv, 1e-8)
chk("CV score at the optimum", sel_cv$score, py$sel_cv_score, 1e-8)
chk("AICc-optimal bandwidth", sel_aicc$bandwidth, py$sel_aicc, 1e-8)
chk("AICc score at the optimum", sel_aicc$score, py$sel_aicc_score, 1e-8)
sel_ad <- .schab_select_bandwidth(y, X, coords, criterion = "aicc",
                                  adaptive = TRUE)
chk("adaptive neighbour count", sel_ad$bandwidth, py$sel_adaptive, 0)
gs <- .schab_golden_section(function(t) (t - 2.75)^2 + 1, 0, 10, tol = 1e-9)
chk("golden section on a known parabola", c(gs$x, gs$value),
    py$golden_parabola, 1e-9)

cat("\n[MGWR backfitting: mgwr/search.py multi_bw]\n")
Xm <- matrix(py$Xm, nrow = n)
ym <- py$ym
mg <- .schab_mgwr_backfit(ym, Xm, coords, criterion = "aicc", tol = 1e-4,
                          max_iter = 25L)
chk("per-covariate bandwidths", mg$bandwidths, py$mgwr_bws, 1e-8)
chk("initial single-bandwidth GWR", mg$bandwidth_gwr, py$mgwr_bw_gwr, 1e-8)
chk("local coefficients", as.numeric(mg$params), py$mgwr_params, 1e-7)
chk("fitted values", mg$fitted, py$mgwr_fitted, 1e-7)
chk("sweep count", mg$n_iter, py$mgwr_n_iter, 0)
chk("SOC history", mg$score_history, py$mgwr_scores, 1e-7)
chk("standardization scale for y", mg$y_scale, py$mgwr_y_scale, 1e-12)
chk("standardization scales for X", mg$x_scale, py$mgwr_x_scale, 1e-12)
if (!identical(mg$converged, py$mgwr_converged)) {
  cat("  convergence flag                                       FAIL\n")
  FAIL <- c(FAIL, "mgwr converged flag")
} else {
  cat(sprintf("  %-54s PASS  converged=%s\n", "convergence flag",
              mg$converged))
}
if (!identical(mg$at_search_boundary, py$mgwr_boundary)) {
  cat("  boundary diagnostic                                    FAIL\n")
  FAIL <- c(FAIL, "mgwr boundary flag")
} else {
  cat(sprintf("  %-54s PASS  at_boundary=%s\n", "boundary diagnostic",
              mg$at_search_boundary))
}

cat(sprintf("\n%s  %d failed\n", strrep("=", 62), length(FAIL)))
if (length(FAIL)) {
  cat("FAILED:", paste(FAIL, collapse = ", "), "\n")
  quit(status = 1)
}
