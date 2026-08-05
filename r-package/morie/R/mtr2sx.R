# SPDX-License-Identifier: AGPL-3.0-or-later
#' Run the instrumental-variable estimate separately in each sex
#'
#' A genetic instrument can act on the outcome through different biology
#' in men and in women, and pooling then estimates a weighted average of
#' two different causal effects -- a quantity that answers no question.
#' Stratifying costs power but keeps the estimand interpretable, and the
#' difference between the strata is itself the interesting result: a large
#' one is evidence of effect modification, not of a bad instrument.
#'
#' Formula: within each stratum the Wald ratio \code{beta = Gamma/gamma}
#' with \code{Gamma} the instrument-outcome slope and \code{gamma} the
#' instrument-exposure slope; \code{se(beta) = se(Gamma)/|gamma|} to first
#' order. The strata are compared by
#' \code{z = (b1 - b2)/sqrt(se1^2 + se2^2)} -- Burgess, Small and Thompson
#' (2017) Sections 2 and 5.
#'
#' @param y Outcome.
#' @param exposure Exposure the instrument is meant to move.
#' @param instrument Genetic instrument, e.g. an allele count.
#' @param sex Stratum label per observation; exactly two distinct values.
#' @return List with \code{estimate}, \code{se}, \code{strata},
#'   \code{beta_by_stratum}, \code{se_by_stratum}, \code{n_by_stratum},
#'   \code{z_het}, \code{p_het}.
#' @references Burgess, S., Small, D. S. and Thompson, S. G. (2017).
#'   Statistical Methods in Medical Research 26(5):2333-2355.
#'   \doi{10.1177/0962280215597579}.
#' @export
Mtr2sx <- function(y, exposure, instrument, sex) {
  Y <- as.numeric(y); X <- as.numeric(exposure)
  G <- as.numeric(instrument); S <- as.numeric(sex)
  n <- length(Y)
  if (n == 0L) stop("no observations")
  if (length(X) != n || length(G) != n || length(S) != n)
    stop("all inputs must have the same length")
  labels <- sort(unique(S))
  if (length(labels) != 2L)
    stop("sex must take exactly two distinct values")
  ratio <- function(g, x, yy) {
    k <- length(g)
    sgg <- sum((g - mean(g))^2)
    if (sgg <= 0) stop("the instrument does not vary in one stratum")
    gx <- sum((g - mean(g)) * (x - mean(x))) / sgg
    gy <- sum((g - mean(g)) * (yy - mean(yy))) / sgg
    if (abs(gx) < 1e-12)
      stop("the instrument does not predict the exposure in one stratum")
    ay <- mean(yy) - gy * mean(g)
    resid <- yy - ay - gy * g
    s2 <- if (k > 2L) sum(resid^2) / (k - 2) else NA_real_
    se_gy <- sqrt(s2 / sgg)
    c(gy / gx, abs(se_gy / gx), k)
  }
  out <- vapply(labels, function(lab) {
    idx <- which(S == lab)
    if (length(idx) < 3L)
      stop("each stratum needs at least three observations")
    ratio(G[idx], X[idx], Y[idx])
  }, numeric(3))
  betas <- out[1, ]; ses <- out[2, ]; ns <- out[3, ]
  w <- 1 / ses^2
  pooled <- sum(w * betas) / sum(w)
  sd <- sqrt(ses[1]^2 + ses[2]^2)
  z <- (betas[1] - betas[2]) / sd
  .t1_result(estimate = pooled, se = sqrt(1 / sum(w)), strata = labels,
             beta_by_stratum = betas, se_by_stratum = ses,
             n_by_stratum = ns, z_het = z,
             p_het = 2 * (1 - .s03pnorm(abs(z))),
             method = "Sex-stratified Mendelian randomisation")
}
