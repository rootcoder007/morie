# SPDX-License-Identifier: AGPL-3.0-or-later
#' Highest posterior density credible interval
#'
#' Chen and Shao (1999), Monte Carlo estimation of Bayesian credible and
#' HPD intervals, JCGS 8(1), 69-92: with n sorted draws and target
#' coverage 1 - alpha, let j = floor((1 - alpha) n); the HPD interval is
#' the pair (theta_(i), theta_(i+j)) minimising theta_(i+j) - theta_(i).
#' The paper is paywalled; the estimator is quoted in its standard
#' published form.  This is the true HPD region only when the posterior is
#' unimodal -- for a multimodal posterior the region is a union of
#' intervals and this scan returns the shortest single one.  That
#' limitation is stated rather than hidden, and the equal-tailed interval
#' is returned so the two can be compared.
#'
#' @param samples posterior draws.
#' @param alpha 1 - coverage.
#' @return list: estimate, width, lo, hi, eq_lo, eq_hi, n, method.
#' @keywords internal
#' @examples
#' Hpdint(c(1, 2, 2.1, 2.2, 9), 0.2)$lo
#' @export
Hpdint <- function(samples, alpha = 0.05) {
  v <- sort(.s03vec(samples)); n <- length(v); a <- as.numeric(alpha)
  j <- as.integer((1 - a) * n)
  if (j < 1L) j <- 1L
  if (j > n - 1L) j <- n - 1L
  best <- 0L; width <- v[j + 1L] - v[1]
  if (n - j > 1L) for (i in seq_len(n - j - 1L)) {
    w <- v[i + j + 1L] - v[i + 1L]
    if (w < width) { width <- w; best <- i }
  }
  list(estimate = width, width = width, lo = v[best + 1L],
       hi = v[best + j + 1L], eq_lo = .s03quantile7(v, a / 2),
       eq_hi = .s03quantile7(v, 1 - a / 2), n = n,
       method = "Chen and Shao (1999) HPD interval scan; single-interval, valid for a unimodal posterior")
}
