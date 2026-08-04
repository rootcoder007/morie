# SPDX-License-Identifier: AGPL-3.0-or-later
#' Cochran's Q test for heterogeneity.
#'
#' Formula: Q = sum_i w_i (y_i - theta_FE)^2 with w_i = 1/v_i, theta_FE = sum w y / sum w; Q ~ chi2_{k-1}
#'
#' @param yi Effect estimates, one per study.
#' @param vi Their sampling variances.

#' @return List with ``Q``, ``df``, ``p_value``, ``theta_fe``, ``se_fe``, ``weights``, ``k``.
#' @references Cochran (1954), The combination of estimates from different experiments, Biometrics 10:101-129. Not held locally; Q = sum w_i (y_i - theta_FE)^2 on k-1 degrees of freedom is the standard published form and is what metafor's rma() reports as QE.
#' @export
Cochranq <- function(yi, vi) {
  y <- .t1_vec(yi); v <- .t1_vec(vi); k <- length(y)
  if (k != length(v)) stop("yi and vi must be the same length")
  if (any(v <= 0)) stop("variances must be positive")
  w <- 1 / v; sw <- sum(w); th <- sum(w * y) / sw
  Q <- sum(w * (y - th)^2); df <- k - 1
  .t1_result(Q = Q, df = df,
             p_value = if (df > 0) stats::pchisq(Q, df, lower.tail = FALSE) else NA_real_,
             theta_fe = th, se_fe = sqrt(1 / sw), weights = w, k = k,
             method = "Cochran's Q test for heterogeneity")
}
