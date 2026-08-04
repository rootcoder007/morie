# SPDX-License-Identifier: AGPL-3.0-or-later
#' Shared numeric helpers for the s04 long-tail batch
#'
#' Internal only. Mirrors \code{morie.fn._s04core} on the Python side so
#' the two arms can be compared value-for-value. Base R supplies the
#' linear algebra and the quantile rule, so most of this is a naming
#' shim; the parts that are not (fixed-iteration IRLS) are the parts
#' that decide whether cross-language parity holds at all.
#'
#' @name s04_core
#' @keywords internal
NULL

.s4_expit <- function(z) ifelse(z >= 0, 1 / (1 + exp(-z)), exp(z) / (1 + exp(z)))

.s4_logit <- function(p) log(p / (1 - p))

.s4_clip <- function(v, lo, hi) pmin(pmax(v, lo), hi)

.s4_median <- function(x) {
  x <- sort(as.numeric(unlist(x)))
  n <- length(x)
  if (n == 0L) return(NaN)
  m <- n %/% 2L
  if (n %% 2L == 1L) x[m + 1L] else 0.5 * (x[m] + x[m + 1L])
}

## R quantile type 7, spelled out so the two arms cannot drift.
.s4_quantile7 <- function(x, p) {
  x <- sort(as.numeric(unlist(x)))
  n <- length(x)
  if (n == 0L) return(NaN)
  if (n == 1L) return(x[1L])
  h <- (n - 1) * p
  lo <- floor(h)
  hi <- min(lo + 1, n - 1)
  x[lo + 1L] + (h - lo) * (x[hi + 1L] - x[lo + 1L])
}

## 0-based order, ties by original position -- matches the Python arm.
.s4_order <- function(x) order(as.numeric(unlist(x)), seq_along(unlist(x))) - 1L

.s4_rank_avg <- function(x) as.numeric(rank(as.numeric(unlist(x)), ties.method = "average"))

.s4_softmax <- function(v) {
  e <- exp(v - max(v))
  e / sum(e)
}

.s4_glmbin <- function(X, y, iters = 25L, ridge = 1e-8) {
  X <- as.matrix(X); y <- as.numeric(y)
  n <- nrow(X); p <- ncol(X)
  beta <- rep(0, p)
  for (it in seq_len(iters)) {
    eta <- as.numeric(X %*% beta)
    mu <- .s4_expit(eta)
    w <- .s4_clip(mu * (1 - mu), 1e-10, 0.25)
    z <- eta + (y - mu) / w
    A <- crossprod(X, X * w) + diag(ridge, p)
    rhs <- as.numeric(crossprod(X, w * z))
    beta <- as.numeric(solve(A, rhs))
  }
  beta
}

.s4_rbf <- function(X, Z, ell = 1) {
  X <- as.matrix(X); Z <- as.matrix(Z)
  out <- matrix(0, nrow(X), nrow(Z))
  for (i in seq_len(nrow(X))) {
    for (j in seq_len(nrow(Z))) {
      out[i, j] <- exp(-0.5 * sum((X[i, ] - Z[j, ])^2) / (ell * ell))
    }
  }
  out
}

.s4_gppost <- function(K, Ks, Kss, y, noise = 1e-6) {
  K <- as.matrix(K); Ks <- as.matrix(Ks)
  n <- nrow(K)
  A <- K + diag(noise, n)
  alpha <- as.numeric(solve(A, as.numeric(y)))
  mean <- as.numeric(crossprod(Ks, alpha))
  V <- solve(A, Ks)
  vr <- as.numeric(Kss) - colSums(Ks * V)
  list(mean = mean, var = vr)
}

.s4_colstd <- function(X) {
  X <- as.matrix(X)
  n <- nrow(X)
  out <- X * 0
  for (j in seq_len(ncol(X))) {
    col <- X[, j]
    m <- sum(col) / n
    s <- if (n > 1) sqrt(sum((col - m)^2) / (n - 1)) else 0
    out[, j] <- if (s > 0) (col - m) / s else 0
  }
  out
}

.s4_euclid <- function(a, b) sqrt(sum((as.numeric(a) - as.numeric(b))^2))

.s4_sgn <- function(v) ifelse(v >= 0, 1, -1)

## Half-away-from-zero. Deliberately not round(): both languages round
## half to even but disagree about which values are exactly half.
.s4_rnd <- function(v) .s4_sgn(v) * floor(abs(v) + 0.5)

## Thin QR by modified Gram-Schmidt. R diagonal is non-negative by
## construction, so Q is unique and there is no sign convention left for
## the two arms to disagree about (LAPACK and LINPACK differ here).
.s4_qr_mgs <- function(A) {
  A <- as.matrix(A)
  n <- nrow(A); p <- ncol(A)
  Q <- A
  R <- matrix(0, p, p)
  for (j in seq_len(p)) {
    if (j > 1L) for (i in seq_len(j - 1L)) {
      R[i, j] <- sum(Q[, i] * Q[, j])
      Q[, j] <- Q[, j] - R[i, j] * Q[, i]
    }
    R[j, j] <- sqrt(sum(Q[, j]^2))
    d <- if (R[j, j] > 1e-300) R[j, j] else 1e-300
    Q[, j] <- Q[, j] / d
  }
  list(Q = Q, R = R)
}

.s4_rank_first <- function(x) {
  x <- as.numeric(unlist(x))
  o <- order(x, seq_along(x))
  r <- integer(length(x))
  r[o] <- seq_along(x)
  r
}

## Outcome model Y = th0 + th1 a + th2 m + th3 a m + th4'c and mediator
## model M = b0 + b1 a + b2'c; cbar is where the decomposition is read.
.s4_medmodels <- function(Y, A, M, Cc = NULL) {
  Y <- as.numeric(Y); A <- as.numeric(A); M <- as.numeric(M)
  n <- length(Y)
  Cm <- if (is.null(Cc)) NULL else as.matrix(Cc)
  XO <- if (is.null(Cm)) cbind(1, A, M, A * M) else cbind(1, A, M, A * M, Cm)
  XM <- if (is.null(Cm)) cbind(1, A) else cbind(1, A, Cm)
  theta <- .t1_lstsq(XO, Y)$beta
  beta <- .t1_lstsq(XM, M)$beta
  cbar <- if (is.null(Cm)) numeric(0) else colSums(Cm) / n
  list(theta = theta, beta = beta, cbar = cbar)
}

## VanderWeele four-way decomposition from fitted coefficients.
.s4_fourway <- function(theta, beta, cbar, a = 1, astar = 0, m = 0) {
  d <- a - astar
  bc <- beta[1] + beta[2] * astar
  if (length(cbar)) bc <- bc + sum(beta[2 + seq_along(cbar)] * cbar)
  cde <- (theta[2] + theta[4] * m) * d
  intref <- theta[4] * (bc - m) * d
  intmed <- theta[4] * beta[2] * d * d
  pie <- (theta[3] * beta[2] + theta[4] * beta[2] * astar) * d
  list(cde = cde, intref = intref, intmed = intmed, pie = pie,
       te = cde + intref + intmed + pie)
}
