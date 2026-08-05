# SPDX-License-Identifier: AGPL-3.0-or-later
#' Outcome-weighted learning for an optimal treatment regime
#'
#' Zhao et al.'s reformulation is that maximizing the value
#' \code{V(d) = E[Y 1{A = d(X)} / pi(A | X)]} is the same problem as
#' MINIMIZING the weighted misclassification risk
#' \code{E[(Y / pi(A | X)) 1{A != d(X)}]}, a classification problem in
#' which the outcome is the weight and the observed treatment is the
#' label. Any classifier will do; the surrogate here is weighted least
#' squares of the +/-1 treatment label on the covariates, which is the
#' least-squares surrogate of the same risk and keeps the fit
#' deterministic and closed form.
#'
#' The rule is \code{d(x) = sign(x'b)}, mapped back to 0/1. NEGATIVE
#' outcomes break the equivalence, because a negative weight rewards
#' misclassification; the standard fix, shifting \code{Y} by its
#' minimum, is applied and REPORTED rather than done silently, since it
#' changes the value scale but not the argmax.
#'
#' @param y Observed outcome, larger is better.
#' @param D Observed binary treatment, 0/1.
#' @param W Covariates, no intercept column; one is added.
#' @param pi Propensity \code{P(A = D_i | W_i)} per unit, or
#'   \code{NULL} for the marginal randomization probability.
#' @return List with \code{beta}, \code{estimate}, \code{value},
#'   \code{value_all_treated}, \code{value_all_control}, \code{rule},
#'   \code{n_treated_by_rule}, \code{shift}, \code{n}, \code{p}.
#' @references Zhao, Y., Zeng, D., Rush, A. J. & Kosorok, M. R. (2012).
#'   Estimating individualized treatment rules using outcome weighted
#'   learning. Journal of the American Statistical Association,
#'   107(499), 1106-1118. doi:10.1080/01621459.2012.695674
#' @export
Owltrn <- function(y, D, W, pi = NULL) {
  yv <- as.numeric(y)
  n <- length(yv)
  if (n == 0L) stop("Owltrn: y is empty")
  Dv <- as.numeric(D)
  if (length(Dv) != n) stop("Owltrn: y and D have different lengths")
  if (!all(Dv %in% c(0, 1))) stop("Owltrn: D must be binary 0/1")
  Wm <- .t1_cbind1(W)
  if (nrow(Wm) != n) stop("Owltrn: W and y have different lengths")
  p <- ncol(Wm)
  if (is.null(pi)) {
    pt <- sum(Dv) / n
    if (pt <= 0 || pt >= 1) stop("Owltrn: both treatments must be observed")
    pv <- ifelse(Dv == 1, pt, 1 - pt)
  } else {
    pv <- as.numeric(pi)
    if (length(pv) != n) stop("Owltrn: pi and y have different lengths")
    if (any(pv <= 0 | pv > 1)) stop("Owltrn: pi must lie in (0, 1]")
  }
  ymin <- min(yv)
  shift <- if (ymin < 0) -ymin else 0
  ys <- yv + shift
  w <- ys / pv
  lab <- ifelse(Dv == 1, 1, -1)
  A <- crossprod(Wm, Wm * w) + diag(1e-10, p)
  b <- crossprod(Wm, w * lab)
  beta <- as.numeric(solve(A, as.numeric(b)))
  rule <- as.numeric(as.numeric(Wm %*% beta) > 0)
  value <- function(rec) {
    m <- as.numeric(rec == Dv)
    den <- sum(m / pv)
    if (den > 0) sum(ys * m / pv) / den else NaN
  }
  .t1_result(beta = beta, estimate = value(rule), value = value(rule),
             value_all_treated = value(rep(1, n)),
             value_all_control = value(rep(0, n)), rule = rule,
             n_treated_by_rule = sum(rule), shift = shift, n = n, p = p,
             method = "Outcome-weighted learning (Zhao et al. 2012)")
}
