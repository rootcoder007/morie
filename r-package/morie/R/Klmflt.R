# SPDX-License-Identifier: AGPL-3.0-or-later
#' Kalman forward recursion driven by a model specification
#'
#' Identical recursion to the matrix-argument form, but the system is
#' passed as one object with entries F, H, Q, R and optionally x0 and
#' P0 -- the shape a fitted model is usually kept in.
#'
#' Formula: predict x_pred = F x, P_pred = F P F' + Q; update with gain
#'   K = P_pred H' (H P_pred H' + R)^{-1}.
#'
#' @param y Observation matrix, one row per time point.
#' @param model Named list with F, H, Q, R and optionally x0, P0.
#' @return List with \code{estimate}, \code{state}, \code{loglik},
#'   \code{n}, \code{method}.
#' @references Kalman (1960), Transactions of the ASME, Journal of Basic
#'   Engineering 82(1):35-45. \doi{10.1115/1.3662552}
#' @export
Klmflt <- function(y, model) {
  Y <- .s03mat(y)
  n <- nrow(Y); m <- ncol(Y)
  if (n == 0L) stop("kalman_filter: y is empty")
  need <- function(k) {
    if (is.null(model[[k]])) stop(paste("kalman_filter: model is missing entry", k))
    model[[k]]
  }
  Fm <- .s03mat(need("F")); Hm <- .s03mat(need("H"))
  Qm <- .s03mat(need("Q")); Rm <- .s03mat(need("R"))
  d <- nrow(Fm)
  if (ncol(Fm) != d) stop("kalman_filter: F must be square")
  if (nrow(Hm) != m || ncol(Hm) != d) stop("kalman_filter: H must be m x d")
  x <- if (is.null(model$x0)) rep(0, d) else .s03vec(model$x0)
  P <- if (is.null(model$P0)) diag(1, d) else .s03mat(model$P0)
  if (length(x) != d || nrow(P) != d) stop("kalman_filter: x0 and P0 must match the state dimension")
  xs <- list(); ll <- 0
  for (t in seq_len(n)) {
    xpred <- as.numeric(Fm %*% x)
    Ppred <- Fm %*% P %*% t(Fm) + Qm
    HP <- Hm %*% Ppred
    S <- HP %*% t(Hm) + Rm
    v <- Y[t, ] - as.numeric(Hm %*% xpred)
    K <- matrix(0, d, m)
    for (j in seq_len(d)) K[j, ] <- .s03cholsolve(S, HP[, j])
    x <- xpred + as.numeric(K %*% v)
    P <- Ppred - (K %*% Hm) %*% Ppred
    sv <- .s03cholsolve(S, v)
    L <- .s03chol(S)
    ll <- ll - 0.5 * (m * log(2 * pi) + 2 * sum(log(diag(L))) + sum(v * sv))
    xs[[t]] <- x
  }
  .t1_result(estimate = xs[[n]][1], state = xs, loglik = ll, n = n,
             method = "forward predict/update recursion, Kalman (1960)")
}
