# Penalised doubly robust TMLE.
# Sources: Belloni, A. & Chernozhukov, V. (2013) "Least squares after
# model selection in high-dimensional sparse models", Bernoulli 19(2),
# 521-547, doi:10.3150/11-BEJ410; van der Laan, M. J. & Gruber, S.
# (2016) "One-step targeted minimum loss-based estimation based on
# universal least favorable one-dimensional submodels", The
# International Journal of Biostatistics 12(1), 351-378,
# doi:10.1515/ijb-2015-0054; van der Laan, M. J. & Rose, S. (2018)
# Targeted Learning in Data Science, Springer,
# doi:10.1007/978-3-319-65304-4.
#
# Native implementation mirroring Python morie.fn.tmldgp exactly: the
# coordinate-descent lasso, post-lasso refit on the support, and
# targeted maximum likelihood with the fluctuation left unpenalised.

.tmldgp_EPS <- 1e-12

#' .tmldgp_logit
#'
#' A step of the tmldgp_native implementation. Called by \code{morie_penalised_tmle}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param p Coerced to numeric by the body, with \code{as.numeric}.
#' @return A numeric value.
#' @export
.tmldgp_logit <- function(p) {
  q <- min(max(as.numeric(p), 1e-9), 1 - 1e-9)
  log(q / (1 - q))
}

#' .tmldgp_expit
#'
#' A step of the tmldgp_native implementation. Called by \code{morie_penalised_tmle}, \code{morie_shrunk_targeting_unsafe}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x Numeric; combined arithmetically in the body.
#' @return One of two values, depending on the branch taken.
#' @export
.tmldgp_expit <- function(x) {
  # vectorised clamp: the scalar if() errors on any vector input
  xc <- pmax(x, -700)
  1 / (1 + exp(-xc))
}

#' .soft
#'
#' A step of the tmldgp_native implementation. Called by \code{morie_lasso_path}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x Numeric; passed to \code{abs}.
#' @param t Numeric; combined arithmetically in the body.
#' @return A numeric value.
#' @export
.soft <- function(x, t) sign(x) * pmax(abs(x) - t, 0)

#' Coordinate-descent lasso path
#'
#' Returns the L1-penalised least squares fit at a single
#' \code{lambda}, with intercept and the selected support.
#'
#' @param X Numeric matrix of predictors (without intercept).
#' @param y Numeric response vector.
#' @param lam Non-negative penalty strength.
#' @param iters Maximum coordinate-descent iterations.
#' @tol Convergence tolerance on the maximum coefficient update.
#' @param tol See Usage.
#' @return A list with \code{beta}, \code{intercept}, \code{support},
#'   \code{lambda}.
#' @references Belloni, A. & Chernozhukov, V. (2013).
#' @export
morie_lasso_path <- function(X, y, lam, iters = 500L, tol = 1e-9) {
  rows <- as.matrix(X)
  storage.mode(rows) <- "double"
  t_ <- as.numeric(y)
  n <- nrow(rows)
  p <- ncol(rows)
  if (length(t_) != n) stop("tmldgp: row/outcome length mismatch")
  if (as.numeric(lam) < 0) stop("tmldgp: lambda cannot be negative")
  b <- rep(0, p)
  b0 <- sum(t_) / n
  for (it in seq_len(as.integer(iters))) {
    big <- 0
    for (j in seq_len(p)) {
      r <- numeric(n)
      for (i in seq_len(n)) {
        s <- 0
        for (q in seq_len(p)) if (q != j) s <- s + rows[i, q] * b[q]
        r[i] <- t_[i] - b0 - s
      }
      zj <- sum(rows[, j]^2)
      if (zj < .tmldgp_EPS) next
      sxx <- 0
      for (i in seq_len(n)) sxx <- sxx + rows[i, j] * r[i]
      new <- .soft(sxx / n, as.numeric(lam)) / (zj / n)
      if (abs(new - b[j]) > big) big <- abs(new - b[j])
      b[j] <- new
    }
    s0 <- 0
    for (i in seq_len(n)) {
      s <- 0
      for (q in seq_len(p)) s <- s + rows[i, q] * b[q]
      s0 <- s0 + t_[i] - s
    }
    b0 <- s0 / n
    if (big < as.numeric(tol)) break
  }
  list(beta = b, intercept = b0,
       support = which(abs(b) > 1e-10) - 1L, lambda = as.numeric(lam))
}

#' Refit by OLS on the lasso's selected support
#'
#' @param X Numeric predictor matrix.
#' @param y Numeric response vector.
#' @param lam Lasso penalty used for the selection step.
#' @return A list with \code{support}, \code{coef}, \code{intercept},
#'   \code{predict}, and the underlying lasso \code{beta}.
#' @references Belloni, A. & Chernozhukov, V. (2013).
#' @export
morie_post_lasso <- function(X, y, lam) {
  rows <- as.matrix(X)
  storage.mode(rows) <- "double"
  t_ <- as.numeric(y)
  sel <- morie_lasso_path(rows, t_, lam)
  S <- sel$support
  if (length(S) == 0L) {
    m_ <- mean(t_)
    return(list(support = integer(0), coef = numeric(0),
                intercept = m_, predict = function(row) m_,
                selected_by = "lasso",
                note = "the lasso selected nothing",
                lasso_beta = sel$beta))
  }
  Xs <- rows[, S + 1L, drop = FALSE]
  Xd <- cbind(1, Xs)
  co <- solve(crossprod(Xd), crossprod(Xd, t_))
  predict_row <- function(row) {
    v <- as.numeric(row)
    s <- co[1]
    for (a in seq_along(S)) s <- s + co[1L + a] * v[S[a] + 1L]
    s
  }
  list(support = S, coef = co, intercept = co[1],
       predict = predict_row, lasso_beta = sel$beta,
       selected_by = "lasso, refitted by OLS",
       note = "post-lasso removes the shrinkage bias on the selected coefficients")
}

#' The thing not to do: penalise the fluctuation
#'
#' @param Q,H,Y Numeric vectors of equal length.
#' @param ridge Optional ridge on \code{epsilon}.
#' @return A list with the would-be \code{epsilon}, the updated fit,
#'   and the un-targeted mean score.
#' @export
morie_shrunk_targeting_unsafe <- function(Q, H, Y, ridge = 1.0) {
  q <- as.numeric(Q)
  h <- as.numeric(H)
  y <- as.numeric(Y)
  n <- length(q)
  if (!(length(h) == n && length(y) == n))
    stop("tmldgp: Q, H, Y must be the same length")
  off <- vapply(q, .tmldgp_logit, numeric(1))
  e <- 0
  for (it in seq_len(60L)) {
    p <- .tmldgp_expit(off + e * h)
    gr <- sum(h * (y - p)) - as.numeric(ridge) * e
    he <- sum(h * h * p * (1 - p)) + as.numeric(ridge)
    if (he < 1e-12) break
    e <- e + gr / he
  }
  upd <- .tmldgp_expit(off + e * h)
  list(epsilon = e, Q_star = upd,
       score = sum(h * (y - upd)) / n,
       caveat = "the score equation is NOT solved when the fluctuation is penalised")
}

#' Penalised doubly robust TMLE
#'
#' Both the propensity \code{g} and the outcome regression
#' \code{bar Q} are fitted by post-lasso; the one-dimensional
#' targeting step is unpenalised.
#'
#' @param y Numeric outcome in \code{\[0,1\]}.
#' @param D Numeric treatment vector.
#' @param X Numeric covariate matrix.
#' @param penalty Lasso penalty.
#' @param iters Maximum targeting iterations.
#' @return A list with the estimate, SE, CI, EIC, and support.
#' @references Belloni, A. & Chernozhukov, V. (2013); van der Laan,
#'   M. J. & Gruber, S. (2016).
#' @export
morie_penalised_tmle <- function(y, D, X, penalty = 0.05, iters = 100) {
  yv <- as.numeric(y)
  a <- as.numeric(D)
  W <- as.matrix(X)
  storage.mode(W) <- "double"
  n <- length(yv)
  if (!(length(a) == nrow(W) && nrow(W) == n))
    stop("tmldgp: the inputs differ in length")
  if (any(yv < 0 | yv > 1))
    stop("tmldgp: the outcome must lie in [0,1]; rescale it first (see tmlcou)")
  gfit <- morie_post_lasso(W, a, penalty)
  gg <- pmin(pmax(vapply(seq_len(n), function(i) gfit$predict(W[i, ]),
                          numeric(1)), 0.02), 0.98)
  Xa <- cbind(a, W)
  qfit <- morie_post_lasso(Xa, yv, penalty)
  q1 <- pmin(pmax(vapply(seq_len(n),
                         function(i) qfit$predict(c(1, W[i, ])),
                         numeric(1)), 1e-6), 1 - 1e-6)
  q0 <- pmin(pmax(vapply(seq_len(n),
                         function(i) qfit$predict(c(0, W[i, ])),
                         numeric(1)), 1e-6), 1 - 1e-6)
  H <- a / gg - (1 - a) / (1 - gg)
  qa <- ifelse(a == 1, q1, q0)
  off <- vapply(qa, .tmldgp_logit, numeric(1))
  e <- 0
  for (it in seq_len(as.integer(iters))) {
    p <- .tmldgp_expit(off + e * H)
    gr <- sum(H * (yv - p))
    he <- sum(H * H * p * (1 - p))
    if (he < 1e-12) break
    step <- gr / he
    e <- e + step
    if (abs(step) < 1e-12) break
  }
  q1s <- .tmldgp_expit(.tmldgp_logit(q1) + e / gg)
  q0s <- .tmldgp_expit(.tmldgp_logit(q0) - e / (1 - gg))
  psi <- mean(q1s - q0s)
  d <- vapply(seq_len(n), function(i) {
    qas <- if (a[i] == 1) q1s[i] else q0s[i]
    H[i] * (yv[i] - qas) + q1s[i] - q0s[i] - psi
  }, numeric(1))
  m <- mean(d)
  se <- sqrt(sum((d - m)^2) / n^2)
  list(estimate = psi, psi = psi, epsilon = e, se = se,
       ci = c(psi - 1.96 * se, psi + 1.96 * se),
       mean_eic = m, solves_eic = abs(m) < 1e-6,
       g_support = gfit$support, Q_support = qfit$support,
       penalty = as.numeric(penalty),
       method = paste("penalised doubly robust TMLE with post-lasso",
                      "nuisance fits; Belloni & Chernozhukov (2013),",
                      "van der Laan & Gruber (2016)"),
       note = paste("the PENALTY is on the nuisances only; penalising",
                    "the fluctuation would break the score equation"))
}

#' Compact one-line summary of the tmldgp recipe
#'
#' @return A character string.
#' @export
morie_tmldgp_cheatsheet <- function() {
  paste("tmldgp: in high dimensions regularise the NUISANCES and",
        "leave the TARGETING alone -- epsilon is one-dimensional and",
        "its MLE is exactly what makes P_n D* = 0, so shrinking it",
        "pulls the estimator back to the untargeted plug-in. Use",
        "POST-LASSO for the nuisance fits: the lasso selects and",
        "shrinks, refitting by OLS on the selection keeps the",
        "selection and undoes the shrinkage. Double robustness",
        "matters MORE under penalisation, since a penalised fit is",
        "deliberately biased and the remainder is a PRODUCT of the",
        "two errors.")
}

morie_penalisedtmle <- morie_penalised_tmle
morie_tmle_doubly_robust_pen <- morie_penalised_tmle

# house entry point: the package exports one morie_<module>
morie_tmldgp <- morie_penalised_tmle
