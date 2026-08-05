# SPDX-License-Identifier: AGPL-3.0-or-later
#' Joint shared-frailty model for recurrent events and a terminal event
#'
#' A single cluster-level frailty w drives both processes:
#' \code{lambda_R(t|w) = w lambda_0R(t)} and
#' \code{lambda_T(t|w) = w^alpha lambda_0T(t)}, with
#' \code{w ~ Gamma(1/theta, 1/theta)} so that E[w] = 1 and Var[w] =
#' theta.  alpha is the association parameter: 0 makes the terminal
#' event independent of the recurrent process, 1 is the ordinary shared
#' frailty, and a negative alpha means clusters with many recurrences
#' die later.  Baselines are constant (exponential).  Because w^alpha
#' appears in the terminal hazard the frailty cannot be integrated out
#' in closed form, so the expectation is a fixed 32-point midpoint rule
#' on the probability scale through the Wilson-Hilferty gamma quantile;
#' maximisation is a fixed-length coordinate golden-section search.
#'
#' Formula: lambda_R(t|w) = w lambda_0R, lambda_T(t|w) = w^alpha lambda_0T.
#'
#' @param time Follow-up time of each record; must be positive.
#' @param event Number of recurrent events for that record.
#' @param terminal Terminal event indicator, 0 or 1.
#' @param cluster Cluster (subject) label.
#' @param sweeps Coordinate-ascent sweeps.
#' @return List with \code{estimate} (alpha), \code{alpha},
#'   \code{theta}, \code{lambda_r}, \code{lambda_t}, \code{loglik},
#'   \code{n_clusters}, \code{n_recurrent}, \code{n_terminal},
#'   \code{exposure}, \code{naive_lambda_r}, \code{naive_lambda_t},
#'   \code{n}, \code{method}.
#' @references Liu, Wolfe and Huang (2004), Shared frailty models for
#'   recurrent events and a terminal event, Biometrics 60(3):747-756.
#'   \doi{10.1111/j.0006-341X.2004.00225.x}
#' @export
Jntfr <- function(time, event, terminal, cluster, sweeps = 4) {
  tv <- .s03vec(time); n <- length(tv)
  if (n == 0L) stop("joint_frailty: time is empty")
  ev <- .s03vec(event); te <- .s03vec(terminal); cl <- .s03vec(cluster)
  if (length(ev) != n || length(te) != n || length(cl) != n)
    stop("joint_frailty: time, event, terminal and cluster have different lengths")
  if (any(tv <= 0)) stop("joint_frailty: time must be positive")
  if (any(te != 0 & te != 1)) stop("joint_frailty: terminal must be 0 or 1")
  if (any(ev < 0)) stop("joint_frailty: event counts must be non-negative")
  labels <- sort(unique(cl)); g <- length(labels)
  N <- numeric(g); A <- numeric(g); dl <- numeric(g); Tt <- numeric(g)
  for (i in seq_len(n)) {
    j <- match(cl[i], labels)
    N[j] <- N[j] + ev[i]; A[j] <- A[j] + tv[i]
    dl[j] <- dl[j] + te[i]; Tt[j] <- Tt[j] + tv[i]
  }
  if (sum(N) <= 0 || sum(dl) <= 0)
    stop("joint_frailty: need at least one recurrent and one terminal event")
  lamR <- sum(N) / sum(A); lamT <- sum(dl) / sum(Tt)
  theta <- 0.5; alpha <- 1
  for (sw in seq_len(as.integer(sweeps))) {
    lamR <- .jntfr_golden(function(v) .jntfr_loglik(v, lamT, theta, alpha, N, A, dl, Tt), 1e-4, 10 * sum(N) / sum(A))
    lamT <- .jntfr_golden(function(v) .jntfr_loglik(lamR, v, theta, alpha, N, A, dl, Tt), 1e-4, 10 * sum(dl) / sum(Tt))
    theta <- .jntfr_golden(function(v) .jntfr_loglik(lamR, lamT, v, alpha, N, A, dl, Tt), 1e-3, 5)
    alpha <- .jntfr_golden(function(v) .jntfr_loglik(lamR, lamT, theta, v, N, A, dl, Tt), -3, 3)
  }
  ll <- .jntfr_loglik(lamR, lamT, theta, alpha, N, A, dl, Tt)
  .t1_result(estimate = alpha, alpha = alpha, theta = theta, lambda_r = lamR,
             lambda_t = lamT, loglik = ll, n_clusters = g,
             n_recurrent = sum(N), n_terminal = sum(dl), exposure = sum(A),
             naive_lambda_r = sum(N) / sum(A), naive_lambda_t = sum(dl) / sum(Tt),
             n = n,
             method = "lambda_R = w lambda_0R, lambda_T = w^alpha lambda_0T, w ~ Gamma(1/theta, 1/theta), Liu, Wolfe & Huang (2004)")
}

#' @keywords internal
#' @noRd
.jntfr_nodes <- function(theta) {
  NQ <- 32L
  k <- 1 / theta
  xs <- numeric(NQ)
  for (i in seq_len(NQ)) {
    u <- (i - 0.5) / NQ
    z <- .s03qnorm(u)
    x <- k * (1 - 1 / (9 * k) + z * sqrt(1 / (9 * k)))^3
    if (x <= 0) x <- 1e-8
    xs[i] <- x / k
  }
  list(x = xs, w = rep(1 / NQ, NQ))
}

#' @keywords internal
#' @noRd
.jntfr_loglik <- function(lamR, lamT, theta, alpha, N, A, dl, Tt) {
  nd <- .jntfr_nodes(theta)
  xs <- nd$x; ws <- nd$w; NQ <- length(xs)
  tot <- 0
  for (i in seq_along(N)) {
    acc <- 0
    for (q in seq_len(NQ)) {
      w <- xs[q]
      lp <- N[i] * log(lamR * w) - lamR * w * A[i]
      wa <- w^alpha
      lp <- lp + dl[i] * log(lamT * wa) - lamT * wa * Tt[i]
      acc <- acc + ws[q] * exp(lp)
    }
    if (acc <= 0) acc <- 1e-300
    tot <- tot + log(acc)
  }
  tot
}

#' @keywords internal
#' @noRd
.jntfr_golden <- function(f, lo, hi, iters = 40L) {
  g <- 0.6180339887498949
  cc <- hi - g * (hi - lo); dd <- lo + g * (hi - lo)
  fc <- f(cc); fd <- f(dd)
  for (i in seq_len(iters)) {
    if (fc > fd) {
      hi <- dd; dd <- cc; fd <- fc
      cc <- hi - g * (hi - lo); fc <- f(cc)
    } else {
      lo <- cc; cc <- dd; fc <- fd
      dd <- lo + g * (hi - lo); fd <- f(dd)
    }
  }
  0.5 * (lo + hi)
}
