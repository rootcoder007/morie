# Synthetic difference in differences.
# Sources: Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W.,
# & Wager, S. (2021) "Synthetic Difference-in-Differences",
# *American Economic Review* 111(12), 4088-4118, for the weighted
# two-way regression of eq. 2.4 and the simplex weight fits of
# eq. 2.8 and 2.9 with a free intercept; the equivalence of the
# three weightings (DID, SC, SDID) and the fact that the intercept
# is what lets SDID match a parallel control path rather than an
# identical one.

#' .causscd_check_grid
#'
#' Part of the causscd_native implementation; see the file header for
#' the source it follows.
#'
#' @param Y See Usage.
#' @param treated See Usage.
#' @param t_post See Usage.
#' @return A list with \code{Y}, \code{n}, \code{T}, \code{tr}, \code{t_post}.
#' @export
.causscd_check_grid <- function(Y, treated, t_post) {
  Y <- as.matrix(Y)
  Y <- storage.mode(Y) <- "double"
  Y <- matrix(as.double(Y), nrow = nrow(Y), ncol = ncol(Y))
  n <- nrow(Y)
  if (n < 2L)
    stop("causscd: need at least two units")
  T <- ncol(Y)
  if (any(!is.finite(Y)))
    stop("causscd: Y contains a non-finite value")
  tr <- as.logical(treated)
  if (length(tr) != n)
    stop("causscd: treated must have one flag per unit")
  t_post <- as.integer(t_post)
  if (!(t_post >= 1L) || !(t_post < T))
    stop("causscd: t_post must lie in 1..T-1 (it is the number of pre-treatment periods)")
  if (!any(tr))
    stop("causscd: no treated units")
  if (all(tr))
    stop("causscd: no control units")
  list(Y = Y, n = n, T = T, tr = tr, t_post = t_post)
}

#' .causscd_project_simplex
#'
#' Part of the causscd_native implementation; see the file header for
#' the source it follows.
#'
#' @param v See Usage.
#' @return The value of \code{pmax}.
#' @export
.causscd_project_simplex <- function(v) {
  m <- length(v)
  u <- sort(v, decreasing = TRUE)
  css <- 0.0
  rho <- 0L
  theta <- 0.0
  if (m > 0L) {
    for (k in seq_len(m)) {
      css <- css + u[k]
      tt <- (css - 1.0) / k
      if (u[k] - tt > 0) {
        rho <- k
        theta <- tt
      }
    }
  }
  pmax(0.0, v - theta)
}

#' .causscd_simplex_fit
#'
#' Part of the causscd_native implementation; see the file header for
#' the source it follows.
#'
#' @param cols See Usage.
#' @param target See Usage.
#' @param penalty See Usage.
#' @param iters Defaults to \code{2000L}.
#' @param tol Defaults to \code{1e-12}.
#' @return A list with \code{w}, \code{icept}.
#' @export
.causscd_simplex_fit <- function(cols, target, penalty, iters = 2000L, tol = 1e-12) {
  m <- length(cols)
  L <- length(target)
  w <- rep(1.0 / m, m)
  step <- NULL
  for (it in seq_len(as.integer(iters))) {
    fit <- numeric(L)
    for (k in seq_len(m))
      fit <- fit + w[k] * cols[[k]]
    icept <- sum(target - fit) / L
    resid <- icept + fit - target
    grad <- numeric(m)
    for (k in seq_len(m))
      grad[k] <- 2.0 * sum(resid * cols[[k]]) + 2.0 * penalty * w[k]
    if (is.null(step)) {
      gnorm <- sqrt(sum(grad * grad))
      if (gnorm == 0) gnorm <- 1.0
      step <- 1.0 / gnorm
    }
    cand <- w - step * grad
    cand <- .causscd_project_simplex(cand)
    if (max(abs(cand - w)) < tol) {
      w <- cand
      break
    }
    w <- cand
  }
  fit <- numeric(L)
  for (k in seq_len(m))
    fit <- fit + w[k] * cols[[k]]
  icept <- sum(target - fit) / L
  list(w = w, icept = icept)
}

#' unit_weights
#'
#' Part of the causscd_native implementation; see the file header for
#' the source it follows.
#'
#' @param Y See Usage.
#' @param treated See Usage.
#' @param t_post See Usage.
#' @param zeta Defaults to \code{NULL}.
#' @return A list with \code{weights}, \code{intercept}, \code{zeta}.
#' @export
unit_weights <- function(Y, treated, t_post, zeta = NULL) {
  g <- .causscd_check_grid(Y, treated, t_post)
  Y <- g$Y; n <- g$n; T <- g$T; tr <- g$tr; t_post <- g$t_post
  co <- which(!tr)
  trt <- which(tr)
  pre <- seq_len(t_post) - 1L
  target <- colSums(Y[trt, pre + 1L, drop = FALSE]) / length(trt)
  cols <- lapply(co, function(i) Y[i, pre + 1L])
  if (is.null(zeta)) {
    diffs <- numeric(0)
    if (t_post >= 2L) {
      for (i in co) {
        for (tt in seq_len(t_post - 1L))
          diffs <- c(diffs, Y[i, tt + 1L] - Y[i, tt])
      }
    }
    if (length(diffs) > 1L) {
      mu <- mean(diffs)
      sd <- sqrt(sum((diffs - mu)^2) / (length(diffs) - 1L))
    } else {
      sd <- 1.0
    }
    zeta <- (length(trt) * (T - t_post)) ^ 0.25 * sd
  }
  sf <- .causscd_simplex_fit(cols, target, (zeta ^ 2) * t_post)
  full <- numeric(n)
  if (length(co) > 0L)
    full[co] <- sf$w
  list(weights = full, intercept = sf$icept, zeta = as.numeric(zeta))
}

#' time_weights
#'
#' Part of the causscd_native implementation; see the file header for
#' the source it follows.
#'
#' @param Y See Usage.
#' @param treated See Usage.
#' @param t_post See Usage.
#' @return A list with \code{weights}, \code{intercept}.
#' @export
time_weights <- function(Y, treated, t_post) {
  g <- .causscd_check_grid(Y, treated, t_post)
  Y <- g$Y; n <- g$n; T <- g$T; tr <- g$tr; t_post <- g$t_post
  co <- which(!tr)
  post <- seq.int(t_post + 1L, T)
  target <- rowSums(Y[co, post, drop = FALSE]) / length(post)
  cols <- lapply(seq_len(t_post), function(tt) Y[co, tt, drop = TRUE])
  sf <- .causscd_simplex_fit(cols, target, 0.0)
  full <- numeric(T)
  if (t_post > 0L)
    full[seq_len(t_post)] <- sf$w
  list(weights = full, intercept = sf$icept)
}

#' sdid
#'
#' Part of the causscd_native implementation; see the file header for
#' the source it follows.
#'
#' @param Y See Usage.
#' @param treated See Usage.
#' @param t_post See Usage.
#' @param method Defaults to \code{"sdid"}.
#' @param zeta Defaults to \code{NULL}.
#' @return A list with \code{estimate}, \code{tau}, \code{unit_weights}, \code{time_weights}, \code{zeta}, \code{delta_treated}, \code{delta_control}, \code{method_name}, \code{n_treated}, \code{n_control}, \code{t_pre}, \code{t_post}, \code{method}, \code{note}.
#' @export
sdid <- function(Y, treated, t_post, method = "sdid", zeta = NULL) {
  g <- .causscd_check_grid(Y, treated, t_post)
  Y <- g$Y; n <- g$n; T <- g$T; tr <- g$tr; t_post <- g$t_post
  if (!(method %in% c("sdid", "did", "sc")))
    stop("causscd: method must be 'sdid', 'did' or 'sc'")
  co <- which(!tr)
  trt <- which(tr)
  pre <- seq_len(t_post) - 1L
  post <- seq.int(t_post + 1L, T) - 1L

  if (method == "did") {
    om <- numeric(n)
    if (length(co) > 0L)
      om[co] <- 1.0 / length(co)
    lam <- numeric(T)
    if (length(pre) > 0L)
      lam[pre + 1L] <- 1.0 / length(pre)
    zeta_used <- 0.0
  } else {
    uw <- unit_weights(Y, treated, t_post, zeta)
    om <- uw$weights
    zeta_used <- uw$zeta
    if (method == "sc") {
      lam <- numeric(T)
      if (length(pre) > 0L)
        lam[pre + 1L] <- 1.0 / length(pre)
    } else {
      tw <- time_weights(Y, treated, t_post)
      lam <- tw$weights
    }
  }

  wavg_pre <- function(i)
    sum(lam[pre + 1L] * Y[i, pre + 1L])
  avg_post <- function(i)
    mean(Y[i, post + 1L])

  delta <- numeric(n)
  for (i in seq_len(n))
    delta[i] <- avg_post(i) - wavg_pre(i)
  d_tr <- mean(delta[trt])
  d_co <- sum(om * delta)
  tau <- d_tr - d_co

  list(estimate = tau,
       tau = tau,
       unit_weights = om,
       time_weights = lam,
       zeta = as.numeric(zeta_used),
       delta_treated = d_tr,
       delta_control = d_co,
       method_name = method,
       n_treated = length(trt),
       n_control = length(co),
       t_pre = t_post,
       t_post = T - t_post,
       method = sprintf("synthetic DID (Arkhangelsky, Athey, Hirshberg, Imbens & Wager 2021), weighting '%s'", method),
       note = paste("all three weightings are the same estimator of eq.",
                    "2.4; DID uses 1/N_co and uniform time weights, SC",
                    "fitted unit weights only, SDID both"))
}

#' morie_causscd
#'
#' Part of the causscd_native implementation; see the file header for
#' the source it follows.
#'
#' @param Y See Usage.
#' @param treated See Usage.
#' @param t_post See Usage.
#' @param zeta Defaults to \code{NULL}.
#' @return The value of \code{p}, as built in the body.
#' @export
morie_causscd <- function(Y, treated, t_post, zeta = NULL) {
  out <- sdid(Y, treated, t_post, "sdid", zeta)
  p <- out
  p$did <- sdid(Y, treated, t_post, "did")$tau
  p$sc <- sdid(Y, treated, t_post, "sc", zeta)$tau
  p$sdid <- out$tau
  p
}

causscd <- morie_causscd

causal_synthetic_did <- unit_weights

#' .causscd_cheatsheet
#'
#' Part of the causscd_native implementation; see the file header for
#' the source it follows.
#'
#' @return A character value.
#' @export
.causscd_cheatsheet <- function() {
  paste("causscd: synthetic DID (Arkhangelsky et al. 2021). Same",
        "weighted two-way regression as DID, but with unit weights",
        "fitted over the simplex WITH a free intercept (so the",
        "controls need only be parallel to the treated path, not",
        "identical to it) and time weights fitted the same way",
        "transposed. method='did' uses 1/N_co and uniform time",
        "weights; 'sc' uses unit weights only; 'sdid' uses both.")
}
