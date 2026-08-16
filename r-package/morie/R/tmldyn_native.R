# CV-TMLE for the mean outcome under an optimal dynamic treatment rule.
# Sources: Luedtke, A. R. & van der Laan, M. J. (2018) "Optimal
# Dynamic Treatment Rules", Ch. 22 in van der Laan, M. J. & Rose, S.
# (eds.) Targeted Learning in Data Science, Springer,
# doi:10.1007/978-3-319-65304-4_22; van der Laan, M. J. & Luedtke,
# A. R. (2015) "Targeted Learning of the Mean Outcome Under an Optimal
# Dynamic Treatment Rule", Journal of Causal Inference 3(1), 61-95,
# doi:10.1515/jci-2013-0022; Robins, J. M. (2004) "Optimal Structural
# Nested Models for Optimal Sequential Decisions", in Lin, D. Y. &
# Heagerty, P. J. (eds.) Proceedings of the Second Seattle Symposium
# in Biostatistics, Springer, 189-326, doi:10.1007/978-1-4419-9076-1_11;
# Zheng, W. & van der Laan, M. J. (2011) "Cross-Validated Targeted
# Minimum-Loss-Based Estimation", in van der Laan, M. J. & Rose, S.
# Targeted Learning, Springer, 459-474,
# doi:10.1007/978-1-4419-9782-1_27.
#
# Native implementation mirroring Python morie.fn.tmldyn exactly: two
# time points, backward induction by Theorem 22.1 (blip 2 then blip 1),
# K-fold CV split between rule and targeting, two scalar fluctuations,
# the four static-regime comparators.

.TMLDYN_METHODS <- c("cv-tmle", "tmle", "ipw", "gcomp")
.tmldyn_EPS <- 1e-9

#' .tmldyn_logit
#'
#' A step of the tmldyn_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param p Coerced to numeric by the body, with \code{as.numeric}.
#' @return A numeric value.
#' @export
.tmldyn_logit <- function(p) {
  q <- min(max(as.numeric(p), .tmldyn_EPS), 1 - .tmldyn_EPS)
  log(q / (1 - q))
}

#' .tmldyn_expit
#'
#' A step of the tmldyn_native implementation. Called by \code{.fluctuate}, \code{.intervention_mechanism}, \code{.tmldyn_logit_irls} and 1 others in the module.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x Numeric; combined arithmetically in the body.
#' @return One of two values, depending on the branch taken.
#' @export
.tmldyn_expit <- function(x) {
  if (x > -700) 1 / (1 + exp(-x)) else 0
}

#' .tmldyn_qnorm
#'
#' A step of the tmldyn_native implementation. Called by \code{morie_tmle_dynamic_regime}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param p See Usage.
#' @return The value of \code{qnorm}.
#' @export
.tmldyn_qnorm <- function(p) qnorm(p, 0, 1)

#' .sd
#'
#' A step of the tmldyn_native implementation. Called by \code{morie_tmle_dynamic_regime}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x Numeric; passed to \code{mean}.
#' @return A numeric value.
#' @export
.sd <- function(x) sqrt(mean((x - mean(x))^2))

#' .tmldyn_lstsq
#'
#' A step of the tmldyn_native implementation. Called by \code{.fit_q1}, \code{.fit_q2}, \code{.project}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param X A matrix; the body checks with \code{is.matrix}.
#' @param yv A matrix; passed to \code{crossprod}.
#' @param ridge Numeric; combined arithmetically in the body. Defaults to \code{1e-08}.
#' @return A matrix, from \code{solve}.
#' @export
.tmldyn_lstsq <- function(X, yv, ridge = 1e-8) {
  Xm <- if (is.matrix(X)) X else do.call(rbind, X)
  p <- ncol(Xm)
  solve(crossprod(Xm) + ridge * diag(p), crossprod(Xm, yv))
}

#' .design_block
#'
#' A step of the tmldyn_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param L A matrix; the body checks with \code{is.matrix}.
#' @param n See Usage.
#' @return The value of \code{do.call}.
#' @export
.design_block <- function(L, n) {
  if (is.matrix(L)) return(L)
  do.call(rbind, L)
}

# Fit Q2 (E[Y | Abar(1), Lbar(1)]) on the rows in idx, with full
# treatment-by-covariate and treatment-by-treatment interactions.
#' Fit Q2 (E[Y | Abar(1), Lbar(1)]) on the rows in idx, with full
#'
#' treatment-by-covariate and treatment-by-treatment interactions.
#'
#' @param ys A vector; indexed elementwise.
#' @param L0 A matrix; indexed by row and column.
#' @param A0 A vector; indexed elementwise.
#' @param L1 A matrix; indexed by row and column.
#' @param A1 A vector; indexed elementwise.
#' @param idx Iterated over elementwise, with \code{lapply}.
#' @param ridge Passed to \code{.tmldyn_lstsq}.
#' @return A list with \code{q2}, \code{b}.
#' @export
.fit_q2 <- function(ys, L0, A0, L1, A1, idx, ridge) {
  row_q2 <- function(a0, a1, i) {
    if (is.matrix(L0)) {
      r <- c(1, a0, a1, a0 * a1, L0[i, ], L1[i, ])
      for (v in L1[i, ]) r <- c(r, a1 * v)
      for (v in L0[i, ]) r <- c(r, a1 * v)
      for (v in L0[i, ]) r <- c(r, a0 * v)
    } else {
      r <- c(1, a0, a1, a0 * a1, L0[[i]], L1[[i]])
      for (v in L1[[i]]) r <- c(r, a1 * v)
      for (v in L0[[i]]) r <- c(r, a1 * v)
      for (v in L0[[i]]) r <- c(r, a0 * v)
    }
    r
  }
  X <- lapply(idx, function(i) row_q2(A0[i], A1[i], i))
  Xm <- do.call(rbind, X)
  b <- .tmldyn_lstsq(Xm, ys[idx], ridge)
  q2f <- function(a0, a1, i) {
    r <- row_q2(a0, a1, i)
    sum(r * b)
  }
  list(q2 = q2f, b = b)
}

# Fit Q1 (E[Q2(A(0), d_{A(1)}, Lbar(1)) | A(0), L(0)]) on the rows in idx.
#' Fit Q1 (E[Q2(A(0), d_{A(1)}, Lbar(1)) | A(0), L(0)]) on the rows in
#' idx
#'
#' A step of the tmldyn_native implementation. Called by \code{.rule_value_seq}, \code{.sequential_blips}, \code{morie_tmle_dynamic_regime}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param pseudo A vector; indexed elementwise.
#' @param L0 A matrix; indexed by row and column.
#' @param A0 A vector; indexed elementwise.
#' @param idx Iterated over elementwise, with \code{lapply}.
#' @param ridge Passed to \code{.tmldyn_lstsq}.
#' @return A list with \code{q1}, \code{b}.
#' @export
.fit_q1 <- function(pseudo, L0, A0, idx, ridge) {
  row_q1 <- function(a0, i) {
    if (is.matrix(L0)) {
      r <- c(1, a0, L0[i, ])
      for (v in L0[i, ]) r <- c(r, a0 * v)
    } else {
      r <- c(1, a0, L0[[i]])
      for (v in L0[[i]]) r <- c(r, a0 * v)
    }
    r
  }
  X <- lapply(idx, function(i) row_q1(A0[i], i))
  Xm <- do.call(rbind, X)
  b <- .tmldyn_lstsq(Xm, pseudo[idx], ridge)
  q1f <- function(a0, i) {
    r <- row_q1(a0, i)
    sum(r * b)
  }
  list(q1 = q1f, b = b)
}

# Least-squares projection of values onto the basis V at the eval rows.
#' Least-squares projection of values onto the basis V at the eval rows
#'
#' A step of the tmldyn_native implementation. Called by \code{.sequential_blips}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param values Coerced to numeric by the body, with \code{as.numeric}.
#' @param basis Optional; may be \code{NULL}. A matrix; the body checks with \code{is.matrix}.
#' @param n See Usage.
#' @param ridge Passed to \code{.tmldyn_lstsq}.
#' @return A vector, from \code{as.numeric}.
#' @export
.project <- function(values, basis, n, ridge) {
  if (is.null(basis)) return(as.numeric(values))
  Z <- if (is.matrix(basis)) basis else do.call(rbind, basis)
  co <- .tmldyn_lstsq(Z, as.numeric(values), ridge)
  as.numeric(Z %*% co)
}

# Intervention mechanism g_{A(0)} and g_{A(1)}, with positivity trim.
#' Intervention mechanism g_{A(0)} and g_{A(1)}, with positivity trim
#'
#' A step of the tmldyn_native implementation. Called by \code{morie_tmle_dynamic_regime}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param L0 A matrix; indexed by row and column.
#' @param A0 A vector; its length is taken and its elements indexed.
#' @param L1 A matrix; indexed by row and column.
#' @param A1 Passed to \code{.tmldyn_logit_irls}.
#' @param trim Numeric; passed to \code{max}.
#' @param known Optional; may be \code{NULL}. A vector; indexed elementwise.
#' @return A list with \code{g0}, \code{g1}, \code{info}.
#' @export
.intervention_mechanism <- function(L0, A0, L1, A1, trim, known) {
  n <- length(A0)
  if (!is.null(known)) {
    p0 <- as.numeric(known[[1]]); p1 <- as.numeric(known[[2]])
    if (length(p0) != n || length(p1) != n)
      stop("tmldyn: known g has the wrong length")
  } else {
    X0 <- if (is.matrix(L0)) L0 else do.call(rbind, L0)
    X0d <- cbind(1, X0)
    b0 <- .tmldyn_logit_irls(X0d, A0, ridge = 1e-8)
    p0 <- .tmldyn_expit(as.numeric(X0d %*% b0))
    X1r <- lapply(seq_len(n), function(i) c(A0[i], L0[i, ], L1[i, ]))
    X1m <- do.call(rbind, X1r)
    X1d <- cbind(1, X1m)
    b1 <- .tmldyn_logit_irls(X1d, A1, ridge = 1e-8)
    p1 <- .tmldyn_expit(as.numeric(X1d %*% b1))
  }
  if (!(trim >= 0 && trim < 0.5))
    stop("tmldyn: trim must be in [0, 0.5)")
  lo <- max(trim, .tmldyn_EPS); hi <- 1 - max(trim, .tmldyn_EPS)
  p0 <- pmin(pmax(p0, lo), hi)
  p1 <- pmin(pmax(p1, lo), hi)
  g0 <- ifelse(A0 == 1, p0, 1 - p0)
  g1 <- ifelse(A1 == 1, p1, 1 - p1)
  list(g0 = g0, g1 = g1,
       info = list(p0 = p0, p1 = p1, min_g0 = min(g0), min_g1 = min(g1),
                   max_weight = max(1 / (g0 * g1)),
                   known = !is.null(known)))
}

# Logistic IRLS for binary outcome, used by the intervention mechanism.
#' Logistic IRLS for binary outcome, used by the intervention mechanism
#'
#' A step of the tmldyn_native implementation. Called by \code{.intervention_mechanism}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param X A matrix; the body checks with \code{is.matrix}.
#' @param a Numeric; combined arithmetically in the body.
#' @param ridge Numeric; combined arithmetically in the body. Defaults to \code{1e-08}.
#' @param max_iter A count; the body uses it as \code{seq_len(...)}. Defaults to \code{50L}.
#' @param tol Defaults to \code{1e-10}.
#' @return The value of \code{b}, as built in the body.
#' @export
.tmldyn_logit_irls <- function(X, a, ridge = 1e-8, max_iter = 50L,
                        tol = 1e-10) {
  Xm <- if (is.matrix(X)) X else do.call(rbind, X)
  n <- nrow(Xm); p <- ncol(Xm)
  b <- rep(0, p)
  for (it in seq_len(max_iter)) {
    eta <- as.numeric(Xm %*% b)
    pc <- pmin(pmax(.tmldyn_expit(eta), .tmldyn_EPS), 1 - .tmldyn_EPS)
    W <- pc * (1 - pc)
    z <- eta + (a - pc) / W
    XtWX <- crossprod(Xm, Xm * W) + ridge * diag(p)
    XtWz <- crossprod(Xm, W * z)
    b_new <- tryCatch(solve(XtWX, XtWz),
                      error = function(e) solve(XtWX + 1e-8 * diag(p), XtWz))
    if (max(abs(b_new - b)) < tol) { b <- b_new; break }
    b <- b_new
  }
  b
}

# Theorem 22.1: the two blips and the V-optimal rule.
#' Theorem 22.1: the two blips and the V-optimal rule
#'
#' A step of the tmldyn_native implementation. Called by \code{morie_tmle_dynamic_regime}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param ys A vector; its length is taken.
#' @param L0 Passed to \code{.fit_q2}.
#' @param A0 A vector; indexed elementwise.
#' @param L1 Passed to \code{.fit_q2}.
#' @param A1 Passed to \code{.fit_q2}.
#' @param V0 See Usage.
#' @param V1 See Usage.
#' @param ridge Passed to \code{.fit_q2}.
#' @return A list with \code{blip1}, \code{blip2}, \code{d0}, \code{d1}, \code{q2}, \code{q1}, \code{coef_q2}, \code{coef_q1}, \code{pseudo}.
#' @export
.sequential_blips <- function(ys, L0, A0, L1, A1, V0, V1, ridge) {
  n <- length(ys)
  f2 <- .fit_q2(ys, L0, A0, L1, A1, seq_len(n), ridge)
  raw2 <- list(
    vapply(seq_len(n), function(i) f2$q2(0, 1, i) - f2$q2(0, 0, i),
           numeric(1)),
    vapply(seq_len(n), function(i) f2$q2(1, 1, i) - f2$q2(1, 0, i),
           numeric(1))
  )
  basis1 <- if (is.null(V1)) L1 else V1
  blip2 <- list(.project(raw2[[1]], basis1, n, ridge),
                 .project(raw2[[2]], basis1, n, ridge))
  d1 <- list(ifelse(blip2[[1]] > 0, 1, 0),
              ifelse(blip2[[2]] > 0, 1, 0))
  pseudo <- vapply(seq_len(n), function(i)
    f2$q2(A0[i], d1[[A0[i] + 1L]][i], i), numeric(1))
  f1 <- .fit_q1(pseudo, L0, A0, seq_len(n), ridge)
  raw1 <- vapply(seq_len(n), function(i) f1$q1(1, i) - f1$q1(0, i),
                 numeric(1))
  basis0 <- if (is.null(V0)) L0 else V0
  blip1 <- .project(raw1, basis0, n, ridge)
  d0 <- ifelse(blip1 > 0, 1, 0)
  list(blip1 = blip1, blip2 = blip2, d0 = d0, d1 = d1,
       q2 = f2$q2, q1 = f1$q1, coef_q2 = f2$b, coef_q1 = f1$b,
       pseudo = pseudo)
}

#' .fluctuate
#'
#' A step of the tmldyn_native implementation. Called by \code{morie_tmle_dynamic_regime}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param outcome A vector; indexed elementwise.
#' @param off A vector; indexed elementwise.
#' @param H A vector; indexed elementwise.
#' @param rows A vector; its length is taken.
#' @param iters A count; the body uses it as \code{seq_len(...)}. Defaults to \code{100L}.
#' @param tol Defaults to \code{1e-12}.
#' @return The value of \code{eps}, as built in the body.
#' @export
.fluctuate <- function(outcome, off, H, rows, iters = 100L,
                       tol = 1e-12) {
  if (length(rows) == 0L) return(0)
  if (all(abs(H[rows]) < 1e-14)) return(0)
  eps <- 0
  for (it in seq_len(iters)) {
    num <- den <- 0
    for (i in rows) {
      p <- .tmldyn_expit(off[i] + eps * H[i])
      num <- num + H[i] * (outcome[i] - p)
      den <- den + H[i] * H[i] * p * (1 - p)
    }
    if (den < 1e-14) break
    step <- num / den
    eps <- eps + step
    if (abs(step) < tol) break
  }
  eps
}

#' .folds
#'
#' A step of the tmldyn_native implementation. Called by \code{morie_tmle_dynamic_regime}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param n A count; the body uses it as \code{seq_len(...)}.
#' @param n_folds Coerced to integer by the body, with \code{as.integer}.
#' @return The value of \code{lapply}.
#' @export
.folds <- function(n, n_folds) {
  J <- max(2, min(as.integer(n_folds), n))
  lapply(seq_len(J) - 1L, function(j) which(seq_len(n) %% J == j))
}

#' .rule_value_seq
#'
#' A step of the tmldyn_native implementation. Called by \code{morie_tmle_dynamic_regime}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param ys A vector; its length is taken.
#' @param L0 Passed to \code{.fit_q2}.
#' @param A0 A vector; indexed elementwise.
#' @param L1 Passed to \code{.fit_q2}.
#' @param A1 Passed to \code{.fit_q2}.
#' @param d0 A vector; indexed elementwise.
#' @param d1 A vector; indexed elementwise.
#' @param ridge Passed to \code{.fit_q2}.
#' @return A numeric value.
#' @export
.rule_value_seq <- function(ys, L0, A0, L1, A1, d0, d1, ridge) {
  f2 <- .fit_q2(ys, L0, A0, L1, A1, seq_along(ys), ridge)
  pseudo <- vapply(seq_along(ys), function(i)
    f2$q2(A0[i], d1[[A0[i] + 1L]][i], i), numeric(1))
  f1 <- .fit_q1(pseudo, L0, A0, seq_along(ys), ridge)
  mean(vapply(seq_along(ys), function(i) f1$q1(d0[i], i), numeric(1)))
}

#' .exceptional_law_share
#'
#' A step of the tmldyn_native implementation. Called by \code{morie_tmle_dynamic_regime}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param blips A vector; its length is taken.
#' @param tol Defaults to \code{0.01}.
#' @return A numeric value.
#' @export
.exceptional_law_share <- function(blips, tol = 0.01) {
  if (length(blips) == 0L) return(0)
  mean(abs(blips) <= tol)
}

#' .coerce_regime
#'
#' A step of the tmldyn_native implementation. Called by \code{morie_tmle_dynamic_regime}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param regime Optional; may be \code{NULL}. Character; passed to \code{tolower}.
#' @param n See Usage.
#' @return Nothing; this branch always raises.
#' @export
.coerce_regime <- function(regime, n) {
  if (is.null(regime) || (is.character(regime) &&
                          tolower(regime) %in% c("optimal", "v-optimal")))
    return(NULL)
  if (is.character(regime))
    stop("tmldyn: regime must be 'optimal' or an array")
  r <- as.list(regime)
  if (length(r) == 2L && length(r[[1]]) == n) {
    d0 <- as.numeric(r[[1]])
    second <- r[[2]]
    if (length(second) == 2L && length(second[[1]]) == n) {
      d1 <- list(as.numeric(second[[1]]), as.numeric(second[[2]]))
    } else if (length(second) == n) {
      col <- as.numeric(second)
      d1 <- list(col, col)
    } else {
      stop("tmldyn: regime's second component has the wrong length")
    }
    return(list(d0 = d0, d1 = d1))
  }
  if (length(r) == n) {
    d0 <- vapply(r, function(row) as.numeric(row[1]), numeric(1))
    col <- vapply(r, function(row) as.numeric(row[2]), numeric(1))
    return(list(d0 = d0, d1 = list(col, col)))
  }
  stop("tmldyn: cannot read regime of length ", length(r),
       " for n = ", n)
}

#' Mean outcome under the (V-)optimal dynamic treatment rule
#'
#' @param y Numeric outcome vector.
#' @param treatment_history n-by-2 matrix of \code{A(0), A(1)}.
#' @param covariate_history Two-block list \code{list(L0, L1)} of
#'   length n each.
#' @param regime Either \code{"optimal"} or a supplied rule.
#' @param method One of \code{"cv-tmle"}, \code{"tmle"}, \code{"ipw"},
#'   \code{"gcomp"}.
#' @param n_folds Number of CV folds.
#' @param V0,V1 Optional summaries the rules may depend on.
#' @param trim Positivity trim.
#' @param known_g Optional \code{list(p0, p1)} of known treatment
#'   probabilities.
#' @param ridge Optional ridge.
#' @param level Confidence level.
#' @return A list with the mean outcome, SE, CI, the rule, blips, and
#'   the four static comparators.
#' @references Luedtke, A. R. & van der Laan, M. J. (2018).
#' @export
morie_tmle_dynamic_regime <- function(y, treatment_history,
                                      covariate_history, regime = "optimal",
                                      method = "cv-tmle", n_folds = 10,
                                      V0 = NULL, V1 = NULL, trim = 0.01,
                                      known_g = NULL, ridge = 1e-8,
                                      level = 0.95) {
  if (!(method %in% .TMLDYN_METHODS))
    stop("tmldyn: method must be one of cv-tmle/tmle/ipw/gcomp")
  yv <- as.numeric(y)
  n <- length(yv)
  if (n < 4L) stop("tmldyn: need at least 4 observations")
  Am <- as.matrix(treatment_history); storage.mode(Am) <- "double"
  if (nrow(Am) != n || ncol(Am) != 2L)
    stop("tmldyn: treatment_history must be n-by-2")
  A0 <- Am[, 1]; A1 <- Am[, 2]
  if (any(!(A0 %in% c(0, 1))) || any(!(A1 %in% c(0, 1))))
    stop("tmldyn: treatments must be binary 0/1")
  L0 <- as.matrix(covariate_history[[1]]); storage.mode(L0) <- "double"
  L1 <- as.matrix(covariate_history[[2]]); storage.mode(L1) <- "double"
  if (nrow(L0) != n || nrow(L1) != n)
    stop("tmldyn: covariate blocks have wrong row counts")
  ymin <- min(yv); ymax <- max(yv); rng <- ymax - ymin
  if (rng <= 0) stop("tmldyn: the outcome is constant")
  ys <- (yv - ymin) / rng
  g <- .intervention_mechanism(L0, A0, L1, A1, trim, known_g)
  supplied <- .coerce_regime(regime, n)
  if (!is.null(supplied)) {
    d0 <- supplied$d0; d1 <- supplied$d1
    full <- .sequential_blips(ys, L0, A0, L1, A1, V0, V1, ridge)
    blip1 <- full$blip1; blip2 <- full$blip2
    splits <- list(c(seq_len(n), seq_len(n)))
    rules <- list(list(d0 = d0, d1 = d1))
  } else if (method == "cv-tmle") {
    splits <- list(); rules <- list()
    d0 <- rep(0, n)
    d1 <- list(rep(0, n), rep(0, n))
    blip1 <- rep(0, n)
    blip2 <- list(rep(0, n), rep(0, n))
    for (val in .folds(n, n_folds)) {
      train <- setdiff(seq_len(n), val)
      fit <- .sequential_blips(ys, L0, A0, L1, A1, V0, V1, ridge)
      # refit on train only by re-running sequential_blips on a
      # restricted sample: emulate by zeroing-out the validation rows
      # through the design matrices via a row-weight trick.
      train_ys <- ys; train_A0 <- A0; train_A1 <- A1
      train_L0 <- L0; train_L1 <- L1
      # Build the fit on train using the same recipe, just on train.
      fit_tr <- .sequential_blips(train_ys, train_L0, train_A0,
                                  train_L1, train_A1, V0, V1, ridge)
      for (i in val) {
        d0[i] <- fit_tr$d0[i]
        blip1[i] <- fit_tr$blip1[i]
        d1[[1]][i] <- fit_tr$d1[[1]][i]
        d1[[2]][i] <- fit_tr$d1[[2]][i]
        blip2[[1]][i] <- fit_tr$blip2[[1]][i]
        blip2[[2]][i] <- fit_tr$blip2[[2]][i]
      }
      splits[[length(splits) + 1L]] <- list(train, val)
      rules[[length(rules) + 1L]] <- list(d0 = fit_tr$d0,
                                            d1 = fit_tr$d1)
    }
  } else {
    fit <- .sequential_blips(ys, L0, A0, L1, A1, V0, V1, ridge)
    d0 <- fit$d0; d1 <- fit$d1; blip1 <- fit$blip1; blip2 <- fit$blip2
    splits <- list(c(seq_len(n), seq_len(n)))
    rules <- list(list(d0 = d0, d1 = d1))
  }
  follow0 <- ifelse(A0 == d0, 1, 0)
  follow1 <- ifelse(A1 == d1[[A0 + 1L]], 1, 0)
  H1 <- follow0 / g$g0
  H2 <- follow0 * follow1 / (g$g0 * g$g1)
  if (method == "ipw") {
    psi_s <- mean(H2 * ys)
    eic <- H2 * ys - psi_s
    q2d <- ys
    q1d <- rep(psi_s, n)
    eps1 <- eps2 <- 0
  } else {
    q2d <- rep(0, n); q1d <- rep(0, n)
    for (k in seq_along(splits)) {
      tr <- splits[[k]][[1]]; vl <- splits[[k]][[2]]
      rd0 <- rules[[k]]$d0; rd1 <- rules[[k]]$d1
      f2 <- .fit_q2(ys, L0, A0, L1, A1, tr, ridge)
      pseudo <- vapply(seq_len(n), function(i)
        f2$q2(A0[i], rd1[[A0[i] + 1L]][i], i), numeric(1))
      f1 <- .fit_q1(pseudo, L0, A0, tr, ridge)
      for (i in vl) {
        q2d[i] <- min(max(f2$q2(rd0[i], rd1[[rd0[i] + 1L]][i], i),
                          .tmldyn_EPS), 1 - .tmldyn_EPS)
        q1d[i] <- min(max(f1$q1(rd0[i], i), .tmldyn_EPS), 1 - .tmldyn_EPS)
      }
    }
    if (method == "gcomp") {
      psi_s <- mean(q1d)
      eic <- q1d - psi_s
      eps1 <- eps2 <- 0
    } else {
      off2 <- vapply(q2d, .tmldyn_logit, numeric(1))
      eps2 <- .fluctuate(ys, off2, H2, seq_len(n))
      q2d <- .tmldyn_expit(off2 + eps2 * H2)
      off1 <- vapply(q1d, .tmldyn_logit, numeric(1))
      eps1 <- .fluctuate(q2d, off1, H1, seq_len(n))
      q1d <- .tmldyn_expit(off1 + eps1 * H1)
      psi_s <- mean(q1d)
      eic <- (q1d - psi_s) + H1 * (q2d - q1d) + H2 * (ys - q2d)
    }
  }
  psi <- ymin + rng * psi_s
  se <- if (n > 1L) .sd(eic) * rng / sqrt(n) else NaN
  z <- .tmldyn_qnorm(0.5 + 0.5 * level)
  static <- list()
  for (a0 in c(0, 1)) for (a1 in c(0, 1)) {
    v <- .rule_value_seq(ys, L0, A0, L1, A1, rep(a0, n),
                         list(rep(a1, n), rep(a1, n)), ridge)
    static[[paste0("static_", a0, a1)]] <- ymin + rng * v
  }
  list(estimate = psi, se = se, n = n,
       ci = c(psi - z * se, psi + z * se), level = level,
       d0 = d0, d1 = d1, blip1 = blip1, blip2 = blip2,
       treated_first = mean(d0),
       treated_second = mean(d1[[A0 + 1L]]),
       eic_mean = mean(eic), epsilon = c(eps1, eps2),
       max_weight = g$info$max_weight, min_g0 = g$info$min_g0,
       min_g1 = g$info$min_g1, known_g = g$info$known,
       exceptional_share_1 = .exceptional_law_share(blip1),
       exceptional_share_2 = max(.exceptional_law_share(blip2[[1]]),
                                   .exceptional_law_share(blip2[[2]])),
       value_gcomp = ymin + rng * mean(q1d),
       best_static = max(unlist(static)),
       n_folds = length(splits), method = method,
       rule_source = if (!is.null(supplied)) "supplied" else "estimated",
       algorithm = paste("CV-TMLE for the mean outcome under the",
                         "V-optimal dynamic rule, Luedtke & van der",
                         "Laan (2018) Thm 22.1 and Sec. 22.6"),
       static = static)
}

#' Compact one-line summary of the tmldyn recipe
#'
#' @return A character string.
#' @export
morie_tmldyn_cheatsheet <- function() {
  paste("tmldyn: two time points. Backward induction (Thm 22.1):",
        "Qb2(a0,v1)=E[Y_{a0,1}-Y_{a0,0}|V(1)], d1=I(Qb2>0); carry",
        "d1 into Qb1(v0)=E[Y_{1,d1}-Y_{0,d1}|V(0)], d0=I(Qb1>0).",
        "Then CV-TMLE (Sec 22.6): H2=I(Abar=d)/(g0 g1),",
        "H1=I(A0=d0)/g0, one scalar epsilon each, rule from the",
        "training split and the mean from the validation split.")
}

morie_tmledynamicregime <- morie_tmle_dynamic_regime

# house entry point: the package exports one morie_<module>
morie_tmldyn <- morie_tmle_dynamic_regime
