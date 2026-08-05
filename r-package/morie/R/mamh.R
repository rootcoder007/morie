# SPDX-License-Identifier: AGPL-3.0-or-later
#' Pool odds ratios across strata without modelling the strata
#'
#' Fitting a stratum effect per table costs a parameter per table, and
#' with sparse tables the maximum-likelihood estimate is badly biased --
#' the classic Neyman-Scott problem. The Mantel-Haenszel weights sidestep
#' it entirely: they are the weights that make the pooled estimate
#' consistent both when the strata are few and large and when they are
#' many and small, which no likelihood-based weighting achieves at once.
#'
#' Formula: \code{OR_MH = sum(a_i d_i/N_i) / sum(b_i c_i/N_i)}; the
#' standard error of its logarithm is the Robins-Breslow-Greenland
#' expression \code{Var = sum(P R)/(2 (sum R)^2) + sum(P S + Q R)/(2 sum R
#' sum S) + sum(Q S)/(2 (sum S)^2)} with \code{P = (a+d)/N},
#' \code{Q = (b+c)/N}, \code{R = ad/N}, \code{S = bc/N} -- Mantel and
#' Haenszel (1959); Robins, Breslow and Greenland (1986).
#'
#' @param a,b,c,d Per-stratum cells: exposed cases, exposed non-cases,
#'   unexposed cases, unexposed non-cases.
#' @param level Confidence level.
#' @return List with \code{OR_MH}, \code{log_OR}, \code{se_log},
#'   \code{ci}, \code{R}, \code{S}, \code{k}.
#' @references Mantel, N. and Haenszel, W. (1959). Journal of the National
#'   Cancer Institute 22(4):719-748. \doi{10.1093/jnci/22.4.719}. Robins,
#'   J., Breslow, N. and Greenland, S. (1986). Biometrics 42(2):311-323.
#'   \doi{10.2307/2531052}.
#' @export
Mamh <- function(a, b, c, d, level = 0.95) {
  A <- as.numeric(a); B <- as.numeric(b)
  C <- as.numeric(c); D <- as.numeric(d)
  k <- length(A)
  if (k == 0L) stop("no strata")
  if (length(B) != k || length(C) != k || length(D) != k)
    stop("the four cell vectors must have equal length")
  if (any(c(A, B, C, D) < 0)) stop("cell counts must be non-negative")
  n <- A + B + C + D
  if (any(n <= 0)) stop("each stratum needs at least one observation")
  P <- (A + D) / n; Q <- (B + C) / n
  R <- A * D / n; S <- B * C / n
  sR <- sum(R); sS <- sum(S)
  if (sS <= 0 || sR <= 0)
    stop("the pooled odds ratio is not finite for these tables")
  orr <- sR / sS
  v <- sum(P * R) / (2 * sR^2) + sum(P * S + Q * R) / (2 * sR * sS) +
    sum(Q * S) / (2 * sS^2)
  se <- sqrt(v); lor <- log(orr)
  z <- .s03qnorm(1 - (1 - as.numeric(level)) / 2)
  .t1_result(OR_MH = orr, log_OR = lor, se_log = se,
             ci = c(exp(lor - z * se), exp(lor + z * se)),
             R = sR, S = sS, k = k,
             method = "Mantel-Haenszel pooled odds ratio")
}
