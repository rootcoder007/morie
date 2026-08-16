# Bayesian optimisation of an expensive black-box function.
# Sources: Mockus, J. (1975) "On Bayesian methods for seeking the
# extremum", in *Optimization Techniques IFIP Technical Conference*,
# 400-404, for the basic scheme. Snoek, J., Larochelle, H., & Adams,
# R. P. (2012) "Practical Bayesian Optimization of Machine Learning
# Algorithms", *NIPS 25*, arXiv:1206.2944, for the three acquisition
# functions in closed form (Equations 1-3: probability of improvement,
# expected improvement, lower confidence bound) and the ARD Matern 5/2
# kernel of Equation 5 and the ARD squared exponential of Equation 4
# (the latter is the original default; the paper argues against it
# because the sample paths are unrealistically smooth). The
# multi-start projected gradient ascent on the acquisition is the
# paper's own inner loop: the acquisition is cheap and differentiable,
# so it is optimised rather than sampled. Kushner, J. for PI. Srinivas,
# N., Krause, A., Kakade, S. M. & Seeger, M. W. (2010) "Gaussian
# Process Optimization in the Bandit Setting: No Regret and Experimental
# Design", *ICML 2010*, for the LCB acquisition.


# Base R has no erf/erfc; both are pnorm in disguise. Defined here so
# the arm stays base-R only, as the package requires.
#' Base R has no erf/erfc; both are pnorm in disguise. Defined here so
#'
#' the arm stays base-R only, as the package requires.
#'
#' @param x Numeric; combined arithmetically in the body.
#' @return A numeric value.
#' @export
.bayopt_erf <- function(x) 2 * pnorm(x * sqrt(2)) - 1
#' .bayopt_erfc
#'
#' A step of the bayopt_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x Numeric; combined arithmetically in the body.
#' @return A numeric value.
#' @export
.bayopt_erfc <- function(x) 2 * pnorm(-x * sqrt(2))

.KERNELS <- c("matern52", "se")
.ACQ <- c("ei", "pi", "lcb")

#' .bayopt_phi
#'
#' A step of the bayopt_native implementation. Called by \code{acquisition_gradient}, \code{expected_improvement}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param z Numeric; combined arithmetically in the body.
#' @return A numeric value.
#' @export
.bayopt_phi <- function(z) exp(-0.5 * z * z) / sqrt(2 * pi)

#' Pnorm is the standard normal CDF. 2 * pnorm(z) - 1 = .erf(z/sqrt(2))
#'
#' Using pnorm is exactly the closed form below.
#'
#' @param z See Usage.
#' @return A numeric value.
#' @export
.Phi <- function(z) 0.5 * (1.0 + 2 * pnorm(z) - 1.0)
# pnorm is the standard normal CDF. 2 * pnorm(z) - 1 = .erf(z/sqrt(2)).
# Using pnorm is exactly the closed form below.

#' .lengths
#'
#' A step of the bayopt_native implementation. Called by \code{gp_posterior_gradient}, \code{matern52}, \code{squared_exponential}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param ls A vector; its length is taken.
#' @param d A count; the body uses it as \code{rep(...)}.
#' @return The value of \code{out}, as built in the body.
#' @export
.lengths <- function(ls, d) {
  if (length(ls) == 1L) {
    out <- rep(as.numeric(ls), d)
  } else {
    out <- as.numeric(ls)
  }
  if (length(out) != d)
    stop("bayopt: length_scale must be a scalar or one value per dimension")
  if (any(out <= 0))
    stop("bayopt: length scales must be positive")
  out
}

#' .r2
#'
#' A step of the bayopt_native implementation. Called by \code{gp_posterior_gradient}, \code{matern52}, \code{squared_exponential}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param a Numeric; combined arithmetically in the body.
#' @param b Numeric; combined arithmetically in the body.
#' @param ls Numeric; combined arithmetically in the body.
#' @return A numeric value.
#' @export
.r2 <- function(a, b, ls) {
  sum((a - b)^2 / (ls^2))
}

#' matern52
#'
#' A step of the bayopt_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param a A vector; its length is taken.
#' @param b Passed to \code{.r2}.
#' @param amplitude Numeric; combined arithmetically in the body. Defaults to \code{1}.
#' @param length_scale Passed to \code{.lengths}. Defaults to \code{1}.
#' @return A numeric value.
#' @export
matern52 <- function(a, b, amplitude = 1.0, length_scale = 1.0) {
  ls <- .lengths(length_scale, length(a))
  r2 <- .r2(a, b, ls)
  s <- sqrt(5.0 * r2)
  amplitude * (1.0 + s + (5.0 / 3.0) * r2) * exp(-s)
}

#' squared_exponential
#'
#' A step of the bayopt_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param a A vector; its length is taken.
#' @param b Passed to \code{.r2}.
#' @param amplitude Numeric; combined arithmetically in the body. Defaults to \code{1}.
#' @param length_scale Passed to \code{.lengths}. Defaults to \code{1}.
#' @return A numeric value.
#' @export
squared_exponential <- function(a, b, amplitude = 1.0, length_scale = 1.0) {
  ls <- .lengths(length_scale, length(a))
  amplitude * exp(-0.5 * .r2(a, b, ls))
}

#' .dkernel_dr2
#'
#' A step of the bayopt_native implementation. Called by \code{gp_posterior_gradient}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param name Compared against \code{"se"}.
#' @param amplitude Numeric; combined arithmetically in the body.
#' @param r2 Numeric; combined arithmetically in the body.
#' @return A numeric value.
#' @export
.dkernel_dr2 <- function(name, amplitude, r2) {
  if (name == "se")
    return(-0.5 * amplitude * exp(-0.5 * r2))
  s <- sqrt(5.0 * r2)
  -(5.0 / 6.0) * amplitude * (1.0 + s) * exp(-s)
}

#' .kernel
#'
#' A step of the bayopt_native implementation. Called by \code{gp_posterior}, \code{gp_posterior_gradient}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param name Compared against \code{"matern52"}.
#' @return One of two values, depending on the branch taken.
#' @export
.kernel <- function(name) {
  if (!(name %in% .KERNELS))
    stop(sprintf("bayopt: kernel must be one of %s", paste(.KERNELS, collapse=", ")))
  if (name == "matern52") matern52 else squared_exponential
}

#' .chol
#'
#' A step of the bayopt_native implementation. Called by \code{gp_posterior}, \code{gp_posterior_gradient}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param A A matrix; indexed by row and column.
#' @return The value of \code{L}, as built in the body.
#' @export
.chol <- function(A) {
  n <- nrow(A)
  L <- matrix(0.0, n, n)
  for (i in seq_len(n)) {
    for (j in seq_len(i)) {
      s <- A[i, j] - sum(L[i, seq_len(j - 1L)] * L[j, seq_len(j - 1L)])
      if (i == j) {
        if (s <= 0)
          stop("bayopt: the covariance matrix is not positive definite; add noise or spread the design points")
        L[i, j] <- sqrt(s)
      } else {
        L[i, j] <- s / L[j, j]
      }
    }
  }
  L
}

#' .chol_solve
#'
#' A step of the bayopt_native implementation. Called by \code{gp_posterior}, \code{gp_posterior_gradient}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param L A matrix; indexed by row and column.
#' @param b A vector; indexed elementwise.
#' @return The value of \code{x}, as built in the body.
#' @export
.chol_solve <- function(L, b) {
  n <- nrow(L)
  y <- numeric(n)
  for (i in seq_len(n))
    y[i] <- (b[i] - sum(L[i, seq_len(i - 1L)] * y[seq_len(i - 1L)])) / L[i, i]
  x <- numeric(n)
  for (i in n:1L) {
    s <- 0.0
    if (i < n) s <- sum(L[(i + 1L):n, i] * x[(i + 1L):n])
    x[i] <- (y[i] - s) / L[i, i]
  }
  x
}

#' gp_posterior
#'
#' A step of the bayopt_native implementation. Called by \code{bayopt}, \code{maximise_acquisition}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param X Iterated over elementwise, with \code{lapply}.
#' @param y Coerced to numeric by the body, with \code{as.numeric}.
#' @param Xs A vector; its length is taken and its elements indexed.
#' @param kernel Passed to \code{.kernel}. Defaults to \code{"matern52"}.
#' @param amplitude Defaults to \code{1}.
#' @param length_scale Defaults to \code{1}.
#' @param noise Defaults to \code{1e-08}.
#' @param mean Optional; may be \code{NULL}. Coerced to numeric by the body, with \code{as.numeric}.
#' @return A list with \code{mean}, \code{variance}, \code{sd}.
#' @export
gp_posterior <- function(X, y, Xs, kernel = "matern52", amplitude = 1.0,
                         length_scale = 1.0, noise = 1e-8, mean = NULL) {
  rows <- lapply(X, function(r) as.numeric(r))
  if (length(rows) == 0)
    stop("bayopt: no observations")
  d <- length(rows[[1]])
  if (any(sapply(rows, length) != d))
    stop("bayopt: X is ragged")
  ys <- as.numeric(y)
  if (length(ys) != length(rows))
    stop("bayopt: one observation per design point")
  if (noise < 0)
    stop("bayopt: noise must be non-negative")
  k <- .kernel(kernel)
  m <- if (is.null(mean)) sum(ys) / length(ys) else as.numeric(mean)
  n <- length(rows)
  K <- matrix(0.0, n, n)
  for (i in seq_len(n))
    for (j in seq_len(n))
      K[i, j] <- k(rows[[i]], rows[[j]], amplitude, length_scale) +
        (if (i == j) noise else 0.0)
  L <- .chol(K)
  alpha <- .chol_solve(L, ys - m)
  out_m <- numeric(length(Xs))
  out_v <- numeric(length(Xs))
  for (kk in seq_along(Xs)) {
    q <- as.numeric(Xs[[kk]])
    if (length(q) != d)
      stop("bayopt: a query point has the wrong dimension")
    ks <- numeric(n)
    for (i in seq_len(n))
      ks[i] <- k(q, rows[[i]], amplitude, length_scale)
    mu <- m + sum(ks * alpha)
    v <- .chol_solve(L, ks)
    var <- k(q, q, amplitude, length_scale) - sum(ks * v)
    out_m[kk] <- mu
    out_v[kk] <- max(var, 0.0)
  }
  list(mean = out_m, variance = out_v, sd = sqrt(out_v))
}

#' gp_posterior_gradient
#'
#' A step of the bayopt_native implementation. Called by \code{maximise_acquisition}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param X Iterated over elementwise, with \code{lapply}.
#' @param y Coerced to numeric by the body, with \code{as.numeric}.
#' @param xs Coerced to numeric by the body, with \code{as.numeric}.
#' @param kernel Passed to \code{.kernel}. Defaults to \code{"matern52"}.
#' @param amplitude Passed to \code{.dkernel_dr2}. Defaults to \code{1}.
#' @param length_scale Passed to \code{.lengths}. Defaults to \code{1}.
#' @param noise Defaults to \code{1e-08}.
#' @param mean Optional; may be \code{NULL}. Coerced to numeric by the body, with \code{as.numeric}.
#' @return A list with \code{grad_mu}, \code{grad_sd}, \code{mu}, \code{sd}.
#' @export
gp_posterior_gradient <- function(X, y, xs, kernel = "matern52", amplitude = 1.0,
                                  length_scale = 1.0, noise = 1e-8, mean = NULL) {
  rows <- lapply(X, function(r) as.numeric(r))
  ys <- as.numeric(y)
  q <- as.numeric(xs)
  if (length(rows) == 0)
    stop("bayopt: no observations")
  d <- length(rows[[1]])
  if (length(q) != d)
    stop("bayopt: the query point has the wrong dimension")
  if (length(ys) != length(rows))
    stop("bayopt: one observation per design point")
  k <- .kernel(kernel)
  ls <- .lengths(length_scale, d)
  m <- if (is.null(mean)) sum(ys) / length(ys) else as.numeric(mean)
  n <- length(rows)
  K <- matrix(0.0, n, n)
  for (i in seq_len(n))
    for (j in seq_len(n))
      K[i, j] <- k(rows[[i]], rows[[j]], amplitude, length_scale) +
        (if (i == j) noise else 0.0)
  L <- .chol(K)
  alpha <- .chol_solve(L, ys - m)
  ks <- numeric(n)
  for (i in seq_len(n))
    ks[i] <- k(q, rows[[i]], amplitude, length_scale)
  v <- .chol_solve(L, ks)
  mu <- m + sum(ks * alpha)
  var <- max(k(q, q, amplitude, length_scale) - sum(ks * v), 0.0)
  sd <- sqrt(var)
  gmu <- numeric(d); gsd <- numeric(d)
  for (dd in seq_len(d)) {
    dk <- numeric(n)
    for (i in seq_len(n)) {
      r2 <- .r2(q, rows[[i]], ls)
      dr2 <- 2.0 * (q[dd] - rows[[i]][dd]) / (ls[dd]^2)
      dk[i] <- .dkernel_dr2(kernel, amplitude, r2) * dr2
    }
    gmu[dd] <- sum(dk * alpha)
    dvar <- -2.0 * sum(dk * v)
    gsd[dd] <- if (sd > 1e-12) dvar / (2.0 * sd) else 0.0
  }
  list(grad_mu = gmu, grad_sd = gsd, mu = mu, sd = sd)
}

#' acquisition_gradient
#'
#' A step of the bayopt_native implementation. Called by \code{maximise_acquisition}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param gmu A vector; its length is taken.
#' @param gsd Numeric; combined arithmetically in the body.
#' @param mu Numeric; combined arithmetically in the body.
#' @param sd Numeric; combined arithmetically in the body.
#' @param best Numeric; combined arithmetically in the body.
#' @param acq One of \code{"ei"}, \code{"lcb"}. Defaults to \code{"ei"}.
#' @param kappa Numeric; combined arithmetically in the body. Defaults to \code{2}.
#' @param xi Numeric; combined arithmetically in the body. Defaults to \code{0}.
#' @return A numeric value.
#' @export
acquisition_gradient <- function(gmu, gsd, mu, sd, best, acq = "ei",
                                 kappa = 2.0, xi = 0.0) {
  if (!(acq %in% .ACQ))
    stop(sprintf("bayopt: acq must be one of %s", paste(.ACQ, collapse=", ")))
  d <- length(gmu)
  if (acq == "lcb")
    return(-gmu + kappa * gsd)
  if (sd <= 1e-12) return(rep(0.0, d))
  g <- (best - xi - mu) / sd
  if (acq == "ei")
    return(.bayopt_phi(g) * gsd - .Phi(g) * gmu)
  dg <- (-gmu - g * gsd) / sd
  .bayopt_phi(g) * dg
}

#' maximise_acquisition
#'
#' A step of the bayopt_native implementation. Called by \code{bayopt}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param X See Usage.
#' @param y See Usage.
#' @param best See Usage.
#' @param box A vector; its length is taken and its elements indexed.
#' @param acq Defaults to \code{"ei"}.
#' @param kernel Defaults to \code{"matern52"}.
#' @param amplitude Defaults to \code{1}.
#' @param length_scale Defaults to \code{1}.
#' @param noise Defaults to \code{1e-08}.
#' @param kappa Defaults to \code{2}.
#' @param xi Defaults to \code{0}.
#' @param starts Optional; may be \code{NULL}. A vector; its length is taken.
#' @param n_starts Coerced to integer by the body, with \code{as.integer}. Defaults to \code{8}.
#' @param max_iter Coerced to integer by the body, with \code{as.integer}. Defaults to \code{60}.
#' @param tol Defaults to \code{1e-08}.
#' @param rnd Defaults to \code{NULL}.
#' @return A list with \code{x}, \code{acq}, \code{n_starts}, \code{evaluations}.
#' @export
maximise_acquisition <- function(X, y, best, box, acq = "ei", kernel = "matern52",
                                 amplitude = 1.0, length_scale = 1.0, noise = 1e-8,
                                 kappa = 2.0, xi = 0.0, starts = NULL,
                                 n_starts = 8, max_iter = 60, tol = 1e-8,
                                 rnd = NULL) {
  d <- length(box)
  if (is.null(rnd)) {
    st <- 12345L
    rnd <- function() {
      st <<- .ghc_lcg31(st)
      st / as.numeric(2L^31)
    }
  }
  if (is.null(starts)) {
    starts <- lapply(seq_len(as.integer(n_starts)), function(i) {
      sapply(seq_len(d), function(i2) box[[i2]][1] + rnd() * (box[[i2]][2] - box[[i2]][1]))
    })
  }
  if (length(starts) == 0)
    stop("bayopt: no starting points")

  score <- function(pt) {
    p <- gp_posterior(X, y, list(pt), kernel, amplitude, length_scale, noise)
    acquire(p$mean[1], p$sd[1], best, acq, kappa, xi)
  }
  clip <- function(pt) mapply(function(v, b) min(max(v, b[1]), b[2]), pt, box)

  best_pt <- NULL; best_val <- -Inf; evals <- 0L
  for (s0 in starts) {
    pt <- as.numeric(s0)
    pt <- mapply(function(v, b) min(max(v, b[1]), b[2]), pt, box)
    val <- score(pt)
    evals <- evals + 1L
    step <- max(sapply(seq_len(d), function(i) box[[i]][2] - box[[i]][1])) * 0.1
    for (it in seq_len(as.integer(max_iter))) {
      g <- gp_posterior_gradient(X, y, pt, kernel, amplitude, length_scale, noise)
      gacq <- acquisition_gradient(g$grad_mu, g$grad_sd, g$mu, g$sd, best, acq, kappa, xi)
      gn <- sqrt(sum(gacq * gacq))
      if (gn < tol) break
      moved <- FALSE
      t <- step
      for (b_ in seq_len(30L)) {
        cand <- mapply(function(v, gi) v + t * gi / gn, pt, gacq)
        cand <- mapply(function(v, b) min(max(v, b[1]), b[2]), cand, box)
        cval <- score(cand)
        evals <- evals + 1L
        if (cval > val + 1e-15) {
          pt <- cand; val <- cval; step <- t * 1.3; moved <- TRUE
          break
        }
        t <- t * 0.5
      }
      if (!moved) break
    }
    if (val > best_val) {
      best_pt <- pt; best_val <- val
    }
  }
  list(x = best_pt, acq = best_val, n_starts = length(starts), evaluations = evals)
}

#' probability_of_improvement
#'
#' A step of the bayopt_native implementation. Called by \code{acquire}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param mu Numeric; combined arithmetically in the body.
#' @param sd Numeric; combined arithmetically in the body.
#' @param best Numeric; combined arithmetically in the body.
#' @param xi Numeric; combined arithmetically in the body. Defaults to \code{0}.
#' @return The value of \code{.Phi}.
#' @export
probability_of_improvement <- function(mu, sd, best, xi = 0.0) {
  if (sd <= 0) return(0.0)
  .Phi((best - xi - mu) / sd)
}

#' expected_improvement
#'
#' A step of the bayopt_native implementation. Called by \code{acquire}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param mu Numeric; combined arithmetically in the body.
#' @param sd Numeric; combined arithmetically in the body.
#' @param best Numeric; combined arithmetically in the body.
#' @param xi Numeric; combined arithmetically in the body. Defaults to \code{0}.
#' @return A numeric value.
#' @export
expected_improvement <- function(mu, sd, best, xi = 0.0) {
  if (sd <= 0) return(0.0)
  g <- (best - xi - mu) / sd
  sd * (g * .Phi(g) + .bayopt_phi(g))
}

#' lower_confidence_bound
#'
#' A step of the bayopt_native implementation. Called by \code{acquire}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param mu Numeric; combined arithmetically in the body.
#' @param sd Numeric; combined arithmetically in the body.
#' @param kappa Numeric; combined arithmetically in the body. Defaults to \code{2}.
#' @return A numeric value.
#' @export
lower_confidence_bound <- function(mu, sd, kappa = 2.0) {
  mu - kappa * sd
}

#' acquire
#'
#' A step of the bayopt_native implementation. Called by \code{bayopt}, \code{maximise_acquisition}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param mu See Usage.
#' @param sd See Usage.
#' @param best See Usage.
#' @param acq One of \code{"ei"}, \code{"pi"}. Defaults to \code{"ei"}.
#' @param kappa Defaults to \code{2}.
#' @param xi Defaults to \code{0}.
#' @return A numeric value.
#' @export
acquire <- function(mu, sd, best, acq = "ei", kappa = 2.0, xi = 0.0) {
  if (!(acq %in% .ACQ))
    stop(sprintf("bayopt: acq must be one of %s", paste(.ACQ, collapse=", ")))
  if (acq == "ei") return(expected_improvement(mu, sd, best, xi))
  if (acq == "pi") return(probability_of_improvement(mu, sd, best, xi))
  -lower_confidence_bound(mu, sd, kappa)
}

#' bayopt
#'
#' A step of the bayopt_native implementation. Called by \code{morie_bayopt}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param f Iterated over elementwise, with \code{sapply}.
#' @param bounds Iterated over elementwise, with \code{lapply}.
#' @param n_iter Coerced to integer by the body, with \code{as.integer}. Defaults to \code{20}.
#' @param n_init Coerced to integer by the body, with \code{as.integer}. Defaults to \code{5}.
#' @param acq Defaults to \code{"ei"}.
#' @param kernel Defaults to \code{"matern52"}.
#' @param amplitude Defaults to \code{1}.
#' @param length_scale Defaults to \code{1}.
#' @param noise Defaults to \code{1e-08}.
#' @param kappa Defaults to \code{2}.
#' @param xi Defaults to \code{0}.
#' @param n_candidates Coerced to integer by the body, with \code{as.integer}. Defaults to \code{200}.
#' @param seed Coerced to integer by the body, with \code{as.integer}. Defaults to \code{0}.
#' @param X0 Optional; may be \code{NULL}. Iterated over elementwise, with \code{lapply}.
#' @param y0 Optional; may be \code{NULL}. Coerced to numeric by the body, with \code{as.numeric}.
#' @param inner One of \code{"gradient"}, \code{"random"}. Defaults to \code{"gradient"}.
#' @param n_starts Defaults to \code{8}.
#' @return A list with \code{estimate}, \code{x_best}, \code{y_best}, \code{X}, \code{y}, \code{trace}, \code{acq}, \code{kernel}, \code{inner}, \code{n_eval}, \code{method}, \code{note}.
#' @export
bayopt <- function(f, bounds, n_iter = 20, n_init = 5, acq = "ei",
                   kernel = "matern52", amplitude = 1.0,
                   length_scale = 1.0, noise = 1e-8, kappa = 2.0, xi = 0.0,
                   n_candidates = 200, seed = 0, X0 = NULL, y0 = NULL,
                   inner = "gradient", n_starts = 8) {
  if (!(inner %in% c("gradient", "random")))
    stop("bayopt: inner must be 'gradient' or 'random'")
  if (n_starts < 1)
    stop("bayopt: n_starts must be positive")
  if (!(acq %in% .ACQ))
    stop(sprintf("bayopt: acq must be one of %s", paste(.ACQ, collapse=", ")))
  box <- lapply(bounds, function(b) c(as.numeric(b[1]), as.numeric(b[2])))
  if (length(box) == 0)
    stop("bayopt: bounds are empty")
  if (any(sapply(box, function(b) b[1] >= b[2])))
    stop("bayopt: each bound must have lo < hi")
  if (n_iter < 1 || n_candidates < 1)
    stop("bayopt: n_iter and n_candidates must be positive")
  if (is.null(X0) && n_init < 2)
    stop("bayopt: at least two initial points are needed")
  d <- length(box)
  # R's `&` is LOGICAL, not bitwise: as.integer(seed) & 0x7FFFFFFF
  # evaluates to TRUE and coerces to 1, so every seed produced the same
  # stream. The Python arm's `int(seed) & 0x7FFFFFFF` is a mask.
  st <- bitwAnd(as.integer(seed), 2147483647L)
  if (st == 0L) st <- 1L
  # local RNG
  rnd_env <- new.env()
  rnd_env$st <- st
  rnd <- function() {
    rnd_env$st <- .ghc_lcg31(rnd_env$st)
    rnd_env$st / as.numeric(2L^31)
  }
  draw <- function() sapply(seq_len(d), function(i) box[[i]][1] + rnd() * (box[[i]][2] - box[[i]][1]))

  if (!is.null(X0)) {
    X <- lapply(X0, function(r) as.numeric(r))
    if (!is.null(y0)) {
      Y <- as.numeric(y0)
    } else {
      Y <- sapply(X, f)
    }
    if (length(X) != length(Y))
      stop("bayopt: X0 and y0 have different lengths")
  } else {
    X <- lapply(seq_len(as.integer(n_init)), function(i) draw())
    Y <- sapply(X, f)
  }
  trace <- list()
  for (it in seq_len(as.integer(n_iter))) {
    best <- min(Y)
    if (inner == "gradient") {
      got <- maximise_acquisition(X, Y, best, box, acq, kernel,
                                  amplitude, length_scale, noise,
                                  kappa, xi, n_starts = n_starts,
                                  rnd = rnd)
      x_new <- got$x; a_val <- got$acq
    } else {
      cand <- lapply(seq_len(as.integer(n_candidates)), function(i) draw())
      post <- gp_posterior(X, Y, cand, kernel, amplitude, length_scale, noise)
      scores <- sapply(seq_along(cand), function(i) acquire(post$mean[i], post$sd[i], best, acq, kappa, xi))
      k <- which.max(scores)
      x_new <- cand[[k]]; a_val <- scores[k]
    }
    X[[length(X) + 1L]] <- x_new
    Y <- c(Y, as.numeric(f(x_new)))
    trace[[length(trace) + 1L]] <- list(x = x_new, y = Y[length(Y)], acq = a_val, best = min(Y))
  }
  b <- which.min(Y)
  list(
    estimate = X[[b]],
    x_best = X[[b]],
    y_best = Y[b],
    X = X,
    y = Y,
    trace = trace,
    acq = acq,
    kernel = kernel,
    inner = inner,
    n_eval = length(Y),
    method = sprintf("Bayesian optimisation (Mockus 1975; Snoek, Larochelle & Adams 2012) with a %s kernel and the %s acquisition", kernel, acq),
    note = "minimisation throughout, as the paper writes it (x_best = argmin); the acquisition is maximised by multi-start projected gradient ascent on the closed-form gradients, with inner='random' kept as the gradient-free baseline"
  )
}

bayesian_optimization <- bayopt

#' morie_bayopt
#'
#' A step of the bayopt_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param ... Passed through.
#' @return The value of \code{bayopt}.
#' @export
morie_bayopt <- function(...) bayopt(...)
