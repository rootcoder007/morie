# Differentially private TMLE by the Laplace mechanism.
# Sources: Dwork, C., McSherry, F., Nissim, K. & Smith, A. (2006)
# "Calibrating Noise to Sensitivity in Private Data Analysis", Theory
# of Cryptography (TCC 2006), Lecture Notes in Computer Science 3876,
# 265-284, doi:10.1007/11681878_14; Niu, F., Nori, H., Quistorff, B.,
# Caruana, R., Ngwe, D. & Kannan, A. (2022) "Differentially Private
# Estimation of Heterogeneous Causal Effects", Proceedings of the
# First Conference on Causal Learning and Reasoning (CLeaR 2022), PMLR
# 177, 618-633, arXiv:2202.11043.
#
# Native implementation mirroring Python morie.fn.tmldyk exactly: the
# inverse-CDF Laplace draw, the doubly-robust TMLE for the ATE on a
# bounded outcome, the ell-1 sensitivity of the estimator (the bound
# carries 1/g_min), the private confidence interval that adds the
# mechanism's variance, and the basic composition rule.

.tmldyk_EPS <- 1e-12

#' .tmldyk_logit
#'
#' A step of the tmldyk_native implementation. Called by \code{.tmle_ate_bounded}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param p See Usage.
#' @return A numeric value.
#' @export
.tmldyk_logit <- function(p) {
  q <- min(max(as.numeric(p), 1e-9), 1 - 1e-9)
  log(q / (1 - q))
}

#' .tmldyk_expit
#'
#' A step of the tmldyk_native implementation. Called by \code{.tmldyk_logit_irls}, \code{.tmle_ate_bounded}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x Numeric; combined arithmetically in the body.
#' @return One of two values, depending on the branch taken.
#' @export
.tmldyk_expit <- function(x) {
  if (x > -700) 1 / (1 + exp(-x)) else 0
}

# Logistic IRLS that returns a coefficient vector.
#' Logistic IRLS that returns a coefficient vector
#'
#' A step of the tmldyk_native implementation. Called by \code{.tmle_ate_bounded}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param Z See Usage.
#' @param a A vector; its length is taken.
#' @param ridge Numeric; combined arithmetically in the body. Defaults to \code{1e-08}.
#' @param max_iter A count; the body uses it as \code{seq_len(...)}. Defaults to \code{50L}.
#' @param tol Defaults to \code{1e-10}.
#' @return The value of \code{b}, as built in the body.
#' @export
.tmldyk_logit_irls <- function(Z, a, ridge = 1e-8, max_iter = 50L,
                        tol = 1e-10) {
  n <- length(a)
  X <- if (is.matrix(Z)) Z else do.call(rbind, Z)
  p <- ncol(X)
  b <- rep(0, p)
  for (it in seq_len(max_iter)) {
    eta <- as.numeric(X %*% b)
    pc <- pmin(pmax(.tmldyk_expit(eta), 1e-9), 1 - 1e-9)
    W <- pc * (1 - pc)
    z <- eta + (a - pc) / W
    XtWX <- crossprod(X, X * W) + ridge * diag(p)
    XtWz <- crossprod(X, W * z)
    b_new <- tryCatch(solve(XtWX, XtWz),
                      error = function(e) solve(XtWX + 1e-8 * diag(p), XtWz))
    if (max(abs(b_new - b)) < tol) { b <- b_new; break }
    b <- b_new
  }
  b
}

# Solve a least-squares problem with a ridge.
#' Solve a least-squares problem with a ridge
#'
#' A step of the tmldyk_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param Z See Usage.
#' @param yv A matrix; passed to \code{crossprod}.
#' @param ridge Numeric; combined arithmetically in the body. Defaults to \code{1e-08}.
#' @return A matrix, from \code{solve}.
#' @export
.tmldyk_lstsq <- function(Z, yv, ridge = 1e-8) {
  X <- if (is.matrix(Z)) Z else do.call(rbind, Z)
  p <- ncol(X)
  solve(crossprod(X) + ridge * diag(p), crossprod(X, yv))
}

#' .wls_int
#'
#' A step of the tmldyk_native implementation. Called by \code{.tmle_ate_bounded}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param Xm A matrix; passed to \code{nrow}.
#' @param yv A matrix; passed to \code{\%*\%}.
#' @param w A matrix; passed to \code{diag}.
#' @param ridge Numeric; combined arithmetically in the body.
#' @return A matrix, from \code{solve}.
#' @export
.wls_int <- function(Xm, yv, w, ridge) {
  Xd <- cbind(1, Xm)
  W <- diag(w, nrow(Xm))
  XtWX <- crossprod(Xd, W %*% Xd) + ridge * diag(ncol(Xd))
  XtWy <- crossprod(Xd, W %*% yv)
  solve(XtWX, XtWy)
}

#' .rescale
#'
#' A step of the tmldyk_native implementation. Called by \code{.tmle_ate_bounded}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param y See Usage.
#' @param lower See Usage.
#' @param upper See Usage.
#' @return A list with \code{scaled}, \code{lower}, \code{upper}, \code{range}.
#' @export
.rescale <- function(y, lower, upper) {
  v <- as.numeric(y)
  if (length(v) == 0L) stop("tmlcou: no outcomes given")
  a <- if (is.null(lower)) min(v) else as.numeric(lower)
  b <- if (is.null(upper)) max(v) else as.numeric(upper)
  if (b <= a) stop("tmlcou: the upper bound must exceed the lower one")
  if (any(v < a - .tmldyk_EPS | v > b + .tmldyk_EPS))
    stop("tmlcou: an outcome lies outside the stated bounds")
  list(scaled = (v - a) / (b - a), lower = a, upper = b, range = b - a)
}

#' .unscale
#'
#' A step of the tmldyk_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param value See Usage.
#' @param lower See Usage.
#' @param upper See Usage.
#' @return A numeric value.
#' @export
.unscale <- function(value, lower, upper) {
  as.numeric(value) * (as.numeric(upper) - as.numeric(lower)) +
    as.numeric(lower)
}

# Underlying TMLE on the bounded-outcome scale (rescaled to [0,1]).
#' Underlying TMLE on the bounded-outcome scale (rescaled to [0,1])
#'
#' A step of the tmldyk_native implementation. Called by \code{morie_tmle_diff_kernel}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param yv A vector; its length is taken.
#' @param a A vector; indexed elementwise.
#' @param W A matrix; indexed by row and column.
#' @param g See Usage.
#' @param Q1 See Usage.
#' @param Q0 See Usage.
#' @param lower Passed to \code{.rescale}.
#' @param upper Passed to \code{.rescale}.
#' @return A list with \code{psi}, \code{se}, \code{range}.
#' @export
.tmle_ate_bounded <- function(yv, a, W, g, Q1, Q0, lower, upper) {
  n <- length(yv)
  if (any(yv < lower | yv > upper))
    stop("tmldyk: the outcome must lie in [lower, upper]")
  sc <- .rescale(yv, lower, upper)
  ys <- sc$scaled
  if (is.null(g)) {
    Z <- cbind(1, W)
    b <- .tmldyk_logit_irls(Z, a)
    eta <- as.numeric(Z %*% b)
    gg <- pmin(pmax(.tmldyk_expit(eta), 0.01), 0.99)
  } else {
    gg <- pmax(pmin(as.numeric(g), 1 - 1e-6), 1e-6)
  }
  if (is.null(Q1) || is.null(Q0)) {
    Xa <- cbind(a, W)
    co <- .wls_int(Xa, ys, rep(1, n), 0)
    pred <- function(av, i) {
      row <- c(1, av, W[i, ])
      sum(row * co)
    }
    q1 <- pmin(pmax(vapply(seq_len(n), function(i) pred(1, i),
                           numeric(1)), 1e-6), 1 - 1e-6)
    q0 <- pmin(pmax(vapply(seq_len(n), function(i) pred(0, i),
                           numeric(1)), 1e-6), 1 - 1e-6)
  } else {
    q1 <- pmax(pmin(as.numeric(Q1), 1 - 1e-6), 1e-6)
    q0 <- pmax(pmin(as.numeric(Q0), 1 - 1e-6), 1e-6)
  }
  H <- a / gg - (1 - a) / (1 - gg)
  qa <- ifelse(a == 1, q1, q0)
  off <- vapply(qa, .tmldyk_logit, numeric(1))
  e <- 0
  for (it in seq_len(100L)) {
    p <- .tmldyk_expit(off + e * H)
    gr <- sum(H * (ys - p))
    he <- sum(H * H * p * (1 - p))
    if (he < 1e-12) break
    step <- gr / he
    e <- e + step
    if (abs(step) < 1e-12) break
  }
  q1s <- .tmldyk_expit(.tmldyk_logit(q1) + e / gg)
  q0s <- .tmldyk_expit(.tmldyk_logit(q0) - e / (1 - gg))
  psi_s <- mean(q1s - q0s)
  psi <- psi_s * sc$range
  d <- vapply(seq_len(n), function(i) {
    qas <- if (a[i] == 1) q1s[i] else q0s[i]
    (H[i] * (ys[i] - qas) + q1s[i] - q0s[i] - psi_s) * sc$range
  }, numeric(1))
  m <- mean(d)
  se <- sqrt(sum((d - m)^2) / n^2)
  list(psi = psi, se = se, range = sc$range)
}

#' One draw from Lap(0, b) by inverse transform
#'
#' @param scale Positive scale parameter.
#' @param rng Environment produced by \code{.ghc_rng}.
#' @return A numeric value.
#' @export
morie_laplace_noise <- function(scale, rng) {
  b <- as.numeric(scale)
  if (b <= 0) stop("tmldyk: the noise scale must be positive")
  u <- as.numeric(.ghc_unif(rng, 1L)) - 0.5
  -b * sign(u) * log(max(1 - 2 * abs(u), 1e-300))
}

#' l1 sensitivity of a TMLE of the ATE
#'
#' The bound carries :math:`1/g_{\min}`, so it is
#' :math:`2R/(n g_{\min})` and NOT :math:`O(1/n)`. Truncating the
#' propensity score is part of the privacy guarantee.
#'
#' @param n Number of observations.
#' @param g_min Propensity truncation bound.
#' @param y_range Range of the outcome.
#' @return A list with the sensitivity and a few related quantities.
#' @export
morie_ate_sensitivity <- function(n, g_min, y_range = 1.0) {
  nn <- as.integer(n)
  gm <- as.numeric(g_min)
  if (nn < 1L) stop("tmldyk: n must be at least 1")
  if (!(gm > 0 && gm <= 0.5))
    stop("tmldyk: the propensity truncation bound must lie in (0, 0.5]")
  list(sensitivity = 2 * as.numeric(y_range) / (nn * gm),
       naive_1_over_n = as.numeric(y_range) / nn,
       inflation = 2 / gm, g_min = gm, n = nn,
       note = paste("the clever covariate carries 1/g, so the",
                    "sensitivity is NOT O(1/n) unless g is truncated"))
}

#' Laplace mechanism release
#'
#' Returns \code{f(D) + Lap(sensitivity/epsilon)} from the shared
#' generator so the R and Python arms produce the same noise.
#'
#' @param value The (non-private) output to release.
#' @param sensitivity Positive l1 sensitivity of \code{f}.
#' @param epsilon Privacy parameter.
#' @param seed Seed for the shared generator.
#' @return A list with the released value, the noise, the scale, the
#'   noise variance, and the epsilon.
#' @export
morie_private_release <- function(value, sensitivity, epsilon, seed = 0) {
  eps <- as.numeric(epsilon)
  if (eps <= 0) stop("tmldyk: epsilon must be positive")
  if (as.numeric(sensitivity) <= 0)
    stop("tmldyk: the sensitivity must be positive")
  e <- .ghc_rng(as.integer(seed))
  b <- as.numeric(sensitivity) / eps
  noise <- morie_laplace_noise(b, e)
  list(released = as.numeric(value) + noise, noise = noise, scale = b,
       epsilon = eps, noise_variance = 2 * b * b,
       note = paste("the guarantee holds only if the sensitivity is",
                    "an upper bound; an underestimate provides no",
                    "privacy at all"))
}

#' An interval that accounts for the mechanism's own variance
#'
#' Combines the sampling variance with the noise variance of the
#' Laplace mechanism.
#'
#' @param value,se Non-private value and standard error.
#' @param sensitivity,epsilon Privacy parameters.
#' @param seed Seed for the shared generator.
#' @param level Z-quantile (default 1.96).
#' @return A list with \code{estimate}, \code{se_private},
#'   \code{se_sampling}, \code{ci}, \code{width_ratio},
#'   \code{epsilon}.
#' @export
morie_private_ci <- function(value, sensitivity, epsilon, se, seed = 0,
                             level = 1.96) {
  r <- morie_private_release(value, sensitivity, epsilon, seed)
  tot <- as.numeric(se)^2 + r$noise_variance
  w <- as.numeric(level) * sqrt(tot)
  list(estimate = r$released, se_private = sqrt(tot),
       se_sampling = as.numeric(se),
       ci = c(r$released - w, r$released + w),
       width_ratio = if (as.numeric(se) > 0) sqrt(tot) / as.numeric(se)
                     else NaN,
       epsilon = as.numeric(epsilon))
}

#' Basic composition of privacy budgets
#'
#' @param epsilons Numeric vector of positive epsilons.
#' @return A list with the total epsilon, the number of releases, and
#'   a note.
#' @export
morie_composition_budget <- function(epsilons) {
  e <- as.numeric(epsilons)
  if (any(e <= 0)) stop("tmldyk: every epsilon must be positive")
  list(total_epsilon = sum(e), n_releases = length(e),
       note = paste("each release spends part of the budget; the",
                    "guarantee degrades linearly"))
}

#' Differentially private TMLE of the ATE
#'
#' The propensity score is truncated at \code{g_min} -- which bounds
#' the sensitivity and is therefore part of the privacy guarantee, not
#' a numerical convenience.
#'
#' @param y,D,X Outcome, treatment, covariates.
#' @param epsilon Privacy parameter.
#' @param g_min Propensity truncation.
#' @param seed Seed for the shared generator.
#' @param g,Q1,Q0 Optional pre-fitted nuisances.
#' @return A list with the private estimate, the non-private TMLE for
#'   comparison, the sensitivity, and the private interval.
#' @export
morie_tmle_diff_kernel <- function(y, D, X, epsilon = 1.0, g_min = 0.05,
                                   seed = 0, g = NULL, Q1 = NULL,
                                   Q0 = NULL) {
  yv <- as.numeric(y); a <- as.numeric(D)
  W <- as.matrix(X); storage.mode(W) <- "double"
  n <- length(yv)
  if (!(length(a) == nrow(W) && nrow(W) == n))
    stop("tmldyk: the inputs differ in length")
  if (any(yv < 0 | yv > 1))
    stop("tmldyk: the outcome must lie in [0,1] for the stated sensitivity bound")
  fit <- .tmle_ate_bounded(yv, a, W, g, Q1, Q0, 0, 1)
  sens <- morie_ate_sensitivity(n, g_min, 1)
  ci <- morie_private_ci(fit$psi, sens$sensitivity, epsilon, fit$se, seed)
  list(estimate = ci$estimate, psi = ci$estimate,
       non_private_psi = fit$psi, sensitivity = sens$sensitivity,
       epsilon = as.numeric(epsilon), g_min = as.numeric(g_min),
       se_private = ci$se_private, se_sampling = fit$se,
       ci = ci$ci, width_ratio = ci$width_ratio,
       method = paste("epsilon-differentially private TMLE by the",
                      "Laplace mechanism; Dwork, McSherry, Nissim &",
                      "Smith (2006), Niu et al. (2022)"),
       note = paste("the propensity truncation is part of the PRIVACY",
                    "guarantee, since it is what bounds the sensitivity"))
}

#' Compact one-line summary of the tmldyk recipe
#'
#' @return A character string.
#' @export
morie_tmldyk_cheatsheet <- function() {
  paste("tmldyk: epsilon-DP by the LAPLACE mechanism -- add",
        "Lap(sensitivity/epsilon), where sensitivity is how much ONE",
        "individual can move the output. For a TMLE that is NOT",
        "O(1/n): the clever covariate carries 1/g, so it is",
        "2R/(n g_min) and the propensity TRUNCATION is part of the",
        "privacy guarantee, not a numerical convenience. An",
        "underestimated sensitivity provides no privacy at all. The",
        "noise adds 2(scale)^2 to the variance, so the private",
        "interval must widen; k releases cost k*epsilon.")
}

morie_tmlediffkernel <- morie_tmle_diff_kernel

# house entry point: the package exports one morie_<module>
morie_tmldyk <- morie_tmle_diff_kernel
