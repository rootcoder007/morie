# SPDX-License-Identifier: AGPL-3.0-or-later
#' Barrier interior-point solver for smooth inequality-constrained problems
#'
#' The core of the IPOPT algorithm: a barrier subproblem solved
#' approximately for a decreasing sequence of mu, with a Newton step and
#' a backtracking line search that keeps the iterate strictly feasible.
#' Derivatives are central differences with a fixed step, so no
#' symbolic gradient is required and both language arms follow the same
#' trajectory.
#'
#' Formula: minimise f(x) - mu sum_i log(-g_i(x)) with g_i(x) <= 0.
#'
#' @param f Objective function of a numeric vector.
#' @param constraints List of functions g_i with g_i(x) <= 0 feasible.
#' @param x0 Strictly feasible starting point.
#' @param mu0 Initial barrier parameter.
#' @param outer Number of mu reductions.
#' @param inner Newton steps per barrier subproblem.
#' @return List with \code{estimate}, \code{x}, \code{objective},
#'   \code{max_violation}, \code{mu_final}, \code{n}, \code{method}.
#' @references Waechter and Biegler (2006), On the implementation of an
#'   interior-point filter line-search algorithm for large-scale
#'   nonlinear programming, Mathematical Programming 106(1):25-57.
#'   \doi{10.1007/s10107-004-0559-y}
#' @export
Ipfsfa <- function(f, constraints, x0, mu0 = 1, outer = 8, inner = 30) {
  x <- .s03vec(x0)
  n <- length(x)
  if (n == 0L) stop("ipopt_solver: x0 is empty")
  if (!is.function(f)) stop("ipopt_solver: f must be callable")
  cons <- as.list(constraints)
  for (g in cons) {
    if (!is.function(g)) stop("ipopt_solver: every constraint must be callable")
    if (as.numeric(g(x)) >= 0) stop("ipopt_solver: x0 must be strictly feasible")
  }
  mu <- as.numeric(mu0)
  if (mu <= 0) stop("ipopt_solver: mu0 must be positive")
  hh <- 1e-5
  phi <- function(x, mu) {
    v <- as.numeric(f(x))
    for (g in cons) {
      gv <- as.numeric(g(x))
      if (gv >= 0) return(Inf)
      v <- v - mu * log(-gv)
    }
    v
  }
  for (o in seq_len(as.integer(outer))) {
    for (k in seq_len(as.integer(inner))) {
      base <- phi(x, mu)
      g <- numeric(n)
      for (j in seq_len(n)) {
        xp <- x; xm <- x; xp[j] <- xp[j] + hh; xm[j] <- xm[j] - hh
        g[j] <- (phi(xp, mu) - phi(xm, mu)) / (2 * hh)
      }
      H <- matrix(0, n, n)
      for (j in seq_len(n)) {
        xp <- x; xm <- x; xp[j] <- xp[j] + hh; xm[j] <- xm[j] - hh
        H[j, j] <- (phi(xp, mu) - 2 * base + phi(xm, mu)) / (hh * hh)
        if (j < n) for (k2 in seq(j + 1L, n)) {
          xpp <- x; xpm <- x; xmp <- x; xmm <- x
          xpp[j] <- xpp[j] + hh; xpp[k2] <- xpp[k2] + hh
          xpm[j] <- xpm[j] + hh; xpm[k2] <- xpm[k2] - hh
          xmp[j] <- xmp[j] - hh; xmp[k2] <- xmp[k2] + hh
          xmm[j] <- xmm[j] - hh; xmm[k2] <- xmm[k2] - hh
          v <- (phi(xpp, mu) - phi(xpm, mu) - phi(xmp, mu) + phi(xmm, mu)) / (4 * hh * hh)
          H[j, k2] <- v; H[k2, j] <- v
        }
      }
      diag(H) <- diag(H) + 1e-8
      step <- tryCatch(.s03cholsolve(H, -g), error = function(e) -g)
      a <- 1; okstep <- FALSE
      for (i in seq_len(60)) {
        xn <- x + a * step
        if (phi(xn, mu) < base) { x <- xn; okstep <- TRUE; break }
        a <- a / 2
      }
      if (!okstep) break
    }
    mu <- mu * 0.2
  }
  viol <- if (length(cons)) max(vapply(cons, function(g) as.numeric(g(x)), 0)) else -1
  .t1_result(estimate = as.numeric(f(x)), x = x, objective = as.numeric(f(x)),
             max_violation = viol, mu_final = mu, n = n,
             method = "decreasing-mu log-barrier with Newton steps, Waechter & Biegler (2006)")
}
