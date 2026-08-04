# SPDX-License-Identifier: AGPL-3.0-or-later
#' Natural direct and indirect effects, both targeted
#'
#' Two nuisance fits and two fluctuations, one for the outcome and one
#' for the mediator. Targeting both buys the double robustness: a
#' mis-specified mediator model can be rescued by a correct outcome model
#' and the other way round, which a product-of-coefficients estimator
#' cannot do.
#'
#' Formula: \code{NDE = E[Y(1, M_0) - Y(0, M_0)]},
#' \code{NIE = E[Y(1, M_1) - Y(1, M_0)]}.
#'
#' @param Y Outcome.
#' @param X Binary treatment.
#' @param M Mediator.
#' @param Cc Baseline covariates.
#' @return List with \code{estimate}, \code{nie}, \code{total}, \code{se}, \code{eps}, \code{n}.
#' @references Zheng, W. & van der Laan, M. J. (2012). IJB 8(1):1-40.
#' @export
TmlMd <- function(Y, X, M, Cc) {
  yv <- as.numeric(Y); Dv <- as.numeric(X); Mv <- as.numeric(M); n <- length(yv)
  W <- cbind(1, as.matrix(Cc))
  ref <- which(Dv <= 0.5); trt <- which(Dv > 0.5)
  m0b <- .s4_ols(W[ref, , drop = FALSE], Mv[ref])$beta
  m1b <- .s4_ols(W[trt, , drop = FALSE], Mv[trt])$beta
  M0 <- as.numeric(W %*% m0b); M1 <- as.numeric(W %*% m1b)
  r0 <- .s4_tmle(yv, Dv, cbind(W, M0))
  r1 <- .s4_tmle(yv, Dv, cbind(W, M1))
  nde <- r0$psi
  nie <- sum(r1$Q1 - r0$Q1) / n
  .t1_result(estimate = nde, nie = nie, total = nde + nie, se = r0$se,
             eps = r0$eps, n = n,
             method = "TMLE for natural direct and indirect effects")
}
