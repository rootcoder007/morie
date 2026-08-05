# SPDX-License-Identifier: AGPL-3.0-or-later

# Profile negative log-likelihood at (a, b); mu and sigma profiled out.
.ht_nll <- function(xv, yv, a, b) {
  n <- length(xv)
  z <- (yv - a * xv) / xv^b
  m <- sum(z) / n
  s2 <- sum((z - m)^2) / n
  if (s2 <= 0) return(list(f = Inf, mu = m, sd = 0))
  list(f = 0.5 * n * log(s2) + b * sum(log(xv)), mu = m, sd = sqrt(s2))
}

#' Heffernan-Tawn conditional extremes model
#'
#' Formula: Y_j | X = x  =  a_j x + x^(b_j) Z_j   for x > u
#'
#' Fitted by Gaussian likelihood on the exceedance set, with the
#' residual mean and sd profiled out at every (a, b), leaving a
#' two-dimensional bounded search over a in [-1, 1] and b in [0, 1).  A
#' deterministic grid then a golden-section refinement in each
#' coordinate keeps both language arms on the same optimum.
#'
#' @param X An n x 2 matrix; column 1 conditions, on a standard
#'   Laplace or exponential-type scale.
#' @param u Conditioning threshold applied to column 1.
#' @return List with \code{a}, \code{b}, \code{mu_z}, \code{sigma_z},
#'   \code{estimate}, \code{nll}, \code{n_exceed}, \code{n},
#'   \code{method}.
#' @references Heffernan & Tawn (2004), JRSS B 66(3):497-546.
#' @export
Evhpvr <- function(X, u) {
  M <- .s03mat(X)
  n <- nrow(M)
  if (n == 0L) stop("empty input: X has no rows")
  if (ncol(M) != 2L) stop("X must have exactly two columns")
  u <- as.numeric(u)
  keep <- M[, 1] > u
  xv <- M[keep, 1]; yv <- M[keep, 2]
  k <- length(xv)
  if (k < 3L) stop("fewer than three exceedances of u; nothing to fit")
  if (any(xv <= 0))
    stop("the conditioning variable must be positive above u")
  a_lo <- -1; a_hi <- 1; b_lo <- 0; b_hi <- 0.999
  bf <- Inf; a <- 0; b <- 0
  for (i in 0:40) {
    aa <- a_lo + (a_hi - a_lo) * i / 40
    for (j in 0:40) {
      bb <- b_lo + (b_hi - b_lo) * j / 40
      f <- .ht_nll(xv, yv, aa, bb)$f
      if (f < bf) { bf <- f; a <- aa; b <- bb }
    }
  }
  da <- (a_hi - a_lo) / 40
  db <- (b_hi - b_lo) / 40
  gr <- 0.5 * (sqrt(5) - 1)
  for (rep in seq_len(60)) {
    lo <- a - da; hi <- a + da
    for (it in seq_len(40)) {
      cc <- hi - gr * (hi - lo); dd <- lo + gr * (hi - lo)
      if (.ht_nll(xv, yv, cc, b)$f < .ht_nll(xv, yv, dd, b)$f) hi <- dd else lo <- cc
    }
    a <- 0.5 * (lo + hi)
    lo <- max(b - db, b_lo); hi <- min(b + db, b_hi)
    for (it in seq_len(40)) {
      cc <- hi - gr * (hi - lo); dd <- lo + gr * (hi - lo)
      if (.ht_nll(xv, yv, a, cc)$f < .ht_nll(xv, yv, a, dd)$f) hi <- dd else lo <- cc
    }
    b <- 0.5 * (lo + hi)
    da <- da * 0.5
    db <- db * 0.5
  }
  r <- .ht_nll(xv, yv, a, b)
  .t1_result(a = a, b = b, mu_z = r$mu, sigma_z = r$sd, estimate = a,
             nll = r$f, n_exceed = k, n = n,
             method = "Heffernan-Tawn conditional extremes model")
}
