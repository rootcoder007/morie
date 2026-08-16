# SPDX-License-Identifier: AGPL-3.0-or-later

#' Internal helpers shared across the Ghosal Bayesian-nonparametrics suite.
#' Not exported; consumed by ghosal_* callables only.
#' @keywords internal
#' @name ghosal_bnp_helpers
NULL

#' .gh_have
#'
#' Part of the helpers_ghosal_bnp implementation; see the file header
#' for the source it follows.
#'
#' @param pkg See Usage.
#' @return The value of \code{requireNamespace}.
#' @export
.gh_have <- function(pkg) requireNamespace(pkg, quietly = TRUE)

#' .gh_pairwise_sq
#'
#' Part of the helpers_ghosal_bnp implementation; see the file header
#' for the source it follows.
#'
#' @param a See Usage.
#' @param b Defaults to \code{a}.
#' @return A numeric value.
#' @export
.gh_pairwise_sq <- function(a, b = a) {
  outer(rowSums(a^2), rowSums(b^2), "+") - 2 * a %*% t(b)
}

#' .gh_bernstein
#'
#' Part of the helpers_ghosal_bnp implementation; see the file header
#' for the source it follows.
#'
#' @param u See Usage.
#' @param K See Usage.
#' @return The value of \code{B}, as built in the body.
#' @export
.gh_bernstein <- function(u, K) {
  u <- pmin(pmax(u, 1e-12), 1 - 1e-12)
  B <- matrix(0, nrow = length(u), ncol = K)
  for (k in seq_len(K)) {
    B[, k] <- stats::dbeta(u, k, K - k + 1)
  }
  B
}

#' .gh_surv_post
#'
#' Part of the helpers_ghosal_bnp implementation; see the file header
#' for the source it follows.
#'
#' @param t See Usage.
#' @param ev See Usage.
#' @param c See Usage.
#' @param lam0 See Usage.
#' @return A list with \code{times}, \code{S}, \code{H}, \code{dH}, \code{lam0}.
#' @export
.gh_surv_post <- function(t, ev, c, lam0) {
  t <- as.numeric(t)
  n <- length(t)
  if (n == 0) {
    return(NULL)
  }
  if (is.null(ev)) ev <- rep(1L, n)
  if (is.null(lam0)) lam0 <- 1 / max(mean(t), 1e-6)
  uniq <- sort(unique(t))
  Y <- vapply(uniq, function(tk) sum(t >= tk), numeric(1))
  dN <- vapply(uniq, function(tk) sum(t == tk & ev == 1), numeric(1))
  dH0 <- diff(c(0, uniq)) * lam0
  dHp <- (c * dH0 + dN) / (c + Y)
  S <- cumprod(1 - pmin(dHp, 1 - 1e-12))
  list(times = uniq, S = S, H = cumsum(dHp), dH = dHp, lam0 = lam0)
}

#' .gh_haar_dwt
#'
#' Part of the helpers_ghosal_bnp implementation; see the file header
#' for the source it follows.
#'
#' @param y See Usage.
#' @return A list with \code{coeffs}, \code{L}.
#' @export
.gh_haar_dwt <- function(y) {
  L <- 1L
  while (L < length(y)) L <- 2L * L
  if (L > length(y)) y <- c(y, rep(0, L - length(y)))
  coeffs <- list()
  cur <- y
  while (length(cur) > 1L) {
    a <- (cur[seq(1, length(cur), by = 2)] + cur[seq(2, length(cur), by = 2)]) / sqrt(2)
    d <- (cur[seq(1, length(cur), by = 2)] - cur[seq(2, length(cur), by = 2)]) / sqrt(2)
    coeffs[[length(coeffs) + 1L]] <- d
    cur <- a
  }
  coeffs[[length(coeffs) + 1L]] <- cur
  list(coeffs = coeffs, L = L)
}
