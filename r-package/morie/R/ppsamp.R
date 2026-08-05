# SPDX-License-Identifier: AGPL-3.0-or-later
#' PPS inclusion probabilities with the two estimators they support
#'
#' Hansen-Hurwitz is the with-replacement estimator built on the
#' selection probabilities \code{p_i = x_i / X}; Horvitz-Thompson is the
#' without-replacement estimator built on \code{pi_i = n p_i}. A unit
#' whose size would give \code{pi_i > 1} cannot be sampled without
#' replacement at that rate and must be taken with certainty, so that
#' case is an error here rather than a silently truncated probability.
#'
#' Formula: \code{p_i = x_i / sum_k x_k}; \code{pi_i = n p_i};
#' \code{Yhat_HH = (1/n) sum_i y_i / p_i}; \code{Yhat_HT = sum_i y_i / pi_i}.
#'
#' @param y Values for the sampled units.
#' @param size Positive size measure, same length as \code{y}.
#' @param n Sample size, at least 1.
#' @return List with \code{pi}, \code{p}, \code{estimate}, \code{hh_total},
#'   \code{ht_total}, \code{se}, \code{X}, \code{n}.
#' @references Hansen, M. H. & Hurwitz, W. N. (1943). On the theory of
#'   sampling from finite populations. Annals of Mathematical Statistics
#'   14(4):333-362. \doi{10.1214/aoms/1177731356}.
#' @export
Ppsamp <- function(y, size, n) {
  y <- as.numeric(unlist(y)); x <- as.numeric(unlist(size))
  if (length(y) == 0L) stop("Ppsamp: y is empty")
  if (length(x) != length(y)) stop("Ppsamp: size must have one entry per observation")
  if (any(x <= 0)) stop("Ppsamp: sizes must be positive")
  n <- as.integer(n)
  if (n < 1L) stop("Ppsamp: n must be at least 1")
  X <- sum(x); p <- x / X; pi <- n * p
  if (any(pi > 1))
    stop("Ppsamp: an inclusion probability exceeds 1; that unit must be selected with certainty")
  zs <- y / p
  hh <- sum(zs) / n
  ht <- sum(y / pi)
  se <- if (length(y) > 1L) sqrt(sum((zs - hh)^2) / ((length(y) - 1) * n)) else NA_real_
  .t1_result(pi = pi, p = p, estimate = hh, hh_total = hh, ht_total = ht,
             se = se, X = X, n = n,
             method = "PPS: Hansen-Hurwitz and Horvitz-Thompson")
}
