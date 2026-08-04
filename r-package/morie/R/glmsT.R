# SPDX-License-Identifier: AGPL-3.0-or-later
#' Kendall rank-correlation test for monotone trend (Kendall 1938)
#'
#' Source: Kendall, M. G. (1938), A new measure of rank correlation,
#' Biometrika 30, 81-93.  The 1938 paper is paywalled here; the measure
#' is quoted in its standard published form \code{tau = (P - Q)/(P + Q)}
#' with P concordant and Q discordant pairs.  With the first argument
#' time this is the Mann-Kendall trend test, \code{S = P - Q}, null
#' variance
#' \code{Var(S) = (n(n-1)(2n+5) - sum_g u_g(u_g-1)(2u_g+5))/18} over
#' groups of tied x values, and continuity-corrected deviate
#' \code{Z = (S - sign(S))/sqrt(Var(S))}, \code{p = 2(1 - Phi(|Z|))}.
#'
#' @param t Numeric ordering variable, normally time.
#' @param x Numeric series to test.
#' @return list: tau, S, var_S, z, p_value, n_concordant, n_discordant,
#'   n, method.
#' @examples
#' Ktrend(1:20, (1:20)^2)$tau
#' @export
Ktrend <- function(t, x) {
  t <- as.numeric(t)
  x <- as.numeric(x)
  n <- length(t)
  if (length(x) != n) stop("t and x must have the same length")
  if (n < 3) stop("need n >= 3")
  xs <- x[order(t)]
  conc <- 0
  disc <- 0
  for (i in seq_len(n - 1)) {
    dx <- xs[(i + 1):n] - xs[i]
    conc <- conc + sum(dx > 0)
    disc <- disc + sum(dx < 0)
  }
  S <- conc - disc
  tau <- if ((conc + disc) > 0) (conc - disc) / (conc + disc) else NaN
  u <- as.numeric(table(xs))
  tie <- sum(u * (u - 1) * (2 * u + 5))
  varS <- (n * (n - 1) * (2 * n + 5) - tie) / 18
  if (varS <= 0) {
    z <- NaN
    p <- NaN
  } else {
    z <- if (S > 0) (S - 1) / sqrt(varS) else if (S < 0) (S + 1) / sqrt(varS) else 0
    p <- min(1, 2 * stats::pnorm(abs(z), lower.tail = FALSE))
  }
  list(
    tau = tau, S = as.integer(S), var_S = varS, z = z, p_value = p,
    n_concordant = conc, n_discordant = disc, n = n,
    method = paste(
      "Kendall (1938) rank-correlation trend test",
      "(Mann-Kendall normal approximation)"
    )
  )
}
