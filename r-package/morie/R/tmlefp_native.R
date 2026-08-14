# Optimal overlap: which subpopulation is estimable, and how to
# weight it.
# Sources: Crump, R. K., Hotz, V. J., Imbens, G. W. & Mitnik, O. A.
# (2009) "Dealing with limited overlap in estimation of average
# treatment effects", Biometrika 96(1), 187-199. Theorems 5.2-5.4,
# Corollaries 5.1-5.2, Theorem 6.1.
#
# Native implementation mirroring Python morie.fn.tmlefp exactly: the
# conjugate-form alpha_from_gamma, the Theorem 5.2 fixed-point on
# gamma = 2 E[k | k < gamma], the Theorem 5.3 one-sided rule, the
# Theorem 5.4 OWATE weights, and the normalised IPW effect over the
# trimmed subpopulation and the untrimmed sample.

#' Invert gamma = 1 / (alpha (1 - alpha)) for alpha
#'
#' Uses the conjugate form to keep precision when gamma is large.
#'
#' @param gamma Threshold, must be at least 4.
#' @return A numeric value in (0, 1/2].
#' @export
morie_alpha_from_gamma <- function(gamma) {
  gamma <- as.numeric(gamma)
  if (gamma < 4)
    stop("tmlefp: gamma must be at least 4; below that the threshold",
         " 1/(alpha(1-alpha)) = gamma has no root in (0, 1/2]")
  root <- sqrt(1 - 4 / gamma)
  2 / (gamma * (1 + root))
}

#' Theorem 5.2: the variance-minimising trimming threshold
#'
#' Solves gamma = 2 E[k(X) | k(X) < gamma] for gamma by fixed point
#' iteration. Homoskedastic unless conditional variances are supplied.
#'
#' @param pscore Numeric vector of propensity scores in (0, 1).
#' @param sigma2_treated,sigma2_control Optional conditional variances.
#' @param tol,max_iter Convergence tolerance and iteration cap.
#' @return A list with \code{alpha}, \code{gamma}, \code{keep},
#'   \code{trim}, \code{no_trimming}, \code{k}.
#' @references Crump, R. K. et al. (2009). Theorem 5.2.
#' @export
morie_optimal_alpha <- function(pscore, sigma2_treated = NULL,
                                sigma2_control = NULL, tol = 1e-12,
                                max_iter = 200) {
  e <- as.numeric(pscore)
  n <- length(e)
  if (n == 0L) stop("tmlefp: no propensity scores")
  if (any(!(e > 0 & e < 1)))
    stop("tmlefp: propensity scores must lie strictly in (0, 1)")
  if (is.null(sigma2_treated) && is.null(sigma2_control)) {
    k <- 1 / (e * (1 - e))
  } else {
    s1 <- if (is.null(sigma2_treated)) rep(1, n) else as.numeric(sigma2_treated)
    s0 <- if (is.null(sigma2_control)) rep(1, n) else as.numeric(sigma2_control)
    if (length(s1) != n || length(s0) != n)
      stop("tmlefp: one conditional variance per unit")
    if (any(s1 <= 0) || any(s0 <= 0))
      stop("tmlefp: conditional variances must be positive")
    k <- s1 / e + s0 / (1 - e)
  }
  mean_k <- mean(k)
  if (max(k) <= 2 * mean_k) {
    return(list(alpha = 0, gamma = Inf, keep = rep(TRUE, n),
                trim = 0, no_trimming = TRUE, k = k))
  }
  gamma <- 2 * mean_k
  for (it in seq_len(as.integer(max_iter))) {
    sel <- k[k < gamma]
    if (length(sel) == 0L)
      stop("tmlefp: the fixed point excluded every unit; check the propensity scores")
    new <- 2 * sum(sel) / length(sel)
    if (abs(new - gamma) < tol * max(1, abs(gamma))) {
      gamma <- new
      break
    }
    gamma <- new
  }
  keep <- k <= gamma
  homosk <- is.null(sigma2_treated) && is.null(sigma2_control)
  alpha <- if (homosk) morie_alpha_from_gamma(gamma) else NaN
  list(alpha = alpha, gamma = gamma, keep = keep,
       trim = n - sum(keep), no_trimming = FALSE, k = k)
}

#' Theorem 5.3: the one-sided threshold for the effect on the treated
#'
#' Homoskedastic, as in the paper.
#'
#' @param pscore Numeric propensity scores.
#' @param treated Integer 0/1 treatment indicators.
#' @param tol,max_iter Convergence parameters.
#' @return A list with \code{alpha_t}, \code{keep}, \code{trim},
#'   \code{no_trimming}.
#' @references Crump, R. K. et al. (2009). Theorem 5.3.
#' @export
morie_optimal_alpha_att <- function(pscore, treated, tol = 1e-12,
                                    max_iter = 200) {
  e <- as.numeric(pscore); w <- as.integer(treated)
  n <- length(e)
  if (n != length(w)) stop("tmlefp: one treatment indicator per unit")
  if (any(!(e > 0 & e < 1)))
    stop("tmlefp: propensity scores must lie strictly in (0, 1)")
  idx <- which(w == 1L)
  if (length(idx) == 0L) stop("tmlefp: no treated units")
  g <- 1 / (1 - e[idx])
  if (max(1 / (1 - e)) <= 2 * sum(g) / length(g)) {
    return(list(alpha_t = 1, keep = rep(TRUE, n), trim = 0,
                no_trimming = TRUE))
  }
  thr <- 2 * sum(g) / length(g)
  for (it in seq_len(as.integer(max_iter))) {
    sel <- (1 / (1 - e))[idx][(1 / (1 - e))[idx] < thr]
    if (length(sel) == 0L)
      stop("tmlefp: the fixed point excluded every treated unit")
    new <- 2 * sum(sel) / length(sel)
    if (abs(new - thr) < tol * max(1, abs(thr))) { thr <- new; break }
    thr <- new
  }
  alpha_t <- 1 - 1 / thr
  list(alpha_t = alpha_t, keep = e <= alpha_t,
       trim = sum(e > alpha_t), no_trimming = FALSE)
}

#' Theorem 5.4 / Corollary 5.2 OWATE weights
#'
#' @param pscore Numeric propensity scores in (0, 1).
#' @param sigma2_treated,sigma2_control Optional conditional variances.
#' @return A numeric vector of weights, one per observation.
#' @references Crump, R. K. et al. (2009). Theorem 5.4, Corollary 5.2.
#' @export
morie_owate_weights <- function(pscore, sigma2_treated = NULL,
                                sigma2_control = NULL) {
  e <- as.numeric(pscore)
  n <- length(e)
  if (any(!(e > 0 & e < 1)))
    stop("tmlefp: propensity scores must lie strictly in (0, 1)")
  if (is.null(sigma2_treated) && is.null(sigma2_control)) {
    return(e * (1 - e))
  }
  s1 <- if (is.null(sigma2_treated)) rep(1, n) else as.numeric(sigma2_treated)
  s0 <- if (is.null(sigma2_control)) rep(1, n) else as.numeric(sigma2_control)
  1 / (s1 / e + s0 / (1 - e))
}

# Normalised IPW effect over a subpopulation or with given weights.
.ipw_dta <- function(y, w, e, keep, weights) {
  n <- length(y)
  sel <- if (is.null(keep)) seq_len(n) else which(keep)
  if (length(sel) == 0L)
    stop("tmlefp: the selected subpopulation is empty")
  om <- if (is.null(weights)) rep(1, n) else weights
  num1 <- sum(om[sel] * w[sel] * y[sel] / e[sel])
  den1 <- sum(om[sel] * w[sel] / e[sel])
  num0 <- sum(om[sel] * (1 - w[sel]) * y[sel] / (1 - e[sel]))
  den0 <- sum(om[sel] * (1 - w[sel]) / (1 - e[sel]))
  if (den1 <= 0 || den0 <= 0)
    stop("tmlefp: the subpopulation has no treated or no control units")
  list(est = num1 / den1 - num0 / den0, n_kept = length(sel))
}

#' Optimal-overlap estimands and their trimming rules
#'
#' Returns the OSATE (Theorem 5.2 or 5.3), the untrimmed full-sample
#' effect, the OWATE (Theorem 5.4), and the variance bound from
#' Theorem 6.1.
#'
#' @param y Numeric outcome vector.
#' @param treatment Integer 0/1 treatment indicators.
#' @param pscore Numeric propensity scores in (0, 1).
#' @param sigma2_treated,sigma2_control Optional conditional variances.
#' @param estimand One of \code{"ate"} or \code{"att"}.
#' @return A list with the OSATE, full-sample, and OWATE estimates, the
#'   trim rule, and the variance bound.
#' @references Crump, R. K. et al. (2009).
#' @export
morie_tmlefp <- function(y, treatment, pscore,
                          sigma2_treated = NULL, sigma2_control = NULL,
                          estimand = "ate") {
  y <- as.numeric(y); w <- as.integer(treatment); e <- as.numeric(pscore)
  n <- length(y)
  if (!(n == length(w) && length(w) == length(e)))
    stop("tmlefp: y, treatment and pscore must have the same length")
  if (any(!(w %in% c(0L, 1L)))) stop("tmlefp: treatment must be 0 or 1")
  if (!(estimand %in% c("ate", "att")))
    stop("tmlefp: estimand must be 'ate' or 'att'")
  if (estimand == "ate") {
    rule <- morie_optimal_alpha(e, sigma2_treated, sigma2_control)
    keep <- rule$keep; alpha <- rule$alpha; gamma <- rule$gamma
  } else {
    rule <- morie_optimal_alpha_att(e, w)
    keep <- rule$keep; alpha <- rule$alpha_t; gamma <- NaN
  }
  k <- if (is.null(sigma2_treated) && is.null(sigma2_control)) {
    1 / (e * (1 - e))
  } else if (!is.null(rule$k)) {
    rule$k
  } else {
    1 / (e * (1 - e))
  }
  bound <- function(sel) {
    m <- k[sel]
    if (length(m) == 0L) return(Inf)
    q <- length(m) / n
    (sum(m) / length(m)) / q
  }
  ipw <- .ipw_dta(y, w, e, keep, NULL)
  full <- .ipw_dta(y, w, e, NULL, NULL)
  om <- morie_owate_weights(e, sigma2_treated, sigma2_control)
  owate <- .ipw_dta(y, w, e, NULL, om)
  list(estimate = ipw$est, osate = ipw$est, ate_full = full$est,
       owate = owate$est, owate_weights = om,
       alpha = alpha, gamma = gamma, keep = keep, n = n,
       n_kept = ipw$n_kept, n_trimmed = n - ipw$n_kept,
       no_trimming = rule$no_trimming,
       variance_bound = bound(keep),
       variance_bound_full = bound(rep(TRUE, n)),
       estimand = estimand,
       note = paste("the estimand CHANGES with the rule: this is the",
                    "effect for the subpopulation kept, not for the",
                    "whole population (Crump et al. 2009, section 5)"),
       method = paste("optimal-overlap subpopulation and weights",
                      "(Crump, Hotz, Imbens & Mitnik 2009)"))
}

#' Compact one-line summary of the tmlefp recipe
#'
#' @return A character string.
#' @export
morie_tmlefp_cheatsheet <- function() {
  paste("tmlefp: optimal overlap (Crump, Hotz, Imbens & Mitnik 2009).",
        "With propensity scores near 0 or 1 the ATE is barely",
        "estimable, so CHANGE THE ESTIMAND: keep the subpopulation",
        "k(x) = sigma1^2/e + sigma0^2/(1-e) <= 1/(alpha(1-alpha)),",
        "where 1/(alpha(1-alpha)) = 2 E[k | k < that] (Thm 5.2);",
        "homoskedastic, that is alpha <= e <= 1-alpha (Cor 5.1). For",
        "the treated it is one-sided, e <= alpha_t (Thm 5.3). Or drop",
        "the indicator entirely and weight by",
        "omega* = (sigma1^2/e + sigma0^2/(1-e))^{-1} = e(1-e)",
        "(Thm 5.4). No trimming is optimal unless sup k > 2 E[k],",
        "equivalently gamma >= 4.")
}

# Compact alias per ledger/NAMING.md
morie_optimal_overlap <- morie_tmlefp
morie_tmle_effective_pi <- morie_tmlefp
