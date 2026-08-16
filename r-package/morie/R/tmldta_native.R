# Data-adaptive target parameters: honest inference after snooping.
# Sources: Hubbard, A. E., Kennedy, C. J. & van der Laan, M. J. (2018)
# "Data-Adaptive Target Parameters", Ch. 9 in van der Laan, M. J. &
# Rose, S. (eds.) Targeted Learning in Data Science, Springer,
# doi:10.1007/978-3-319-65304-4_9; Hubbard, A. E., Kherad-Pajouh, S.
# & van der Laan, M. J. (2016) "Statistical Inference for Data
# Adaptive Target Parameters", The International Journal of
# Biostatistics 12(1), 3-19, doi:10.1515/ijb-2015-0013; van der Laan,
# M. J. & Luedtke, A. R. (2015) "Targeted Learning of the Mean Outcome
# Under an Optimal Dynamic Treatment Rule", Journal of Causal
# Inference 3(1), 61-95, doi:10.1515/jci-2013-0022.
#
# Native implementation mirroring Python morie.fn.tmldta exactly: V-fold
# parameter-generation / estimation split, logistic Q-surface with
# level-by-W interactions, three-category g normalised, one scalar
# epsilon per split, the per-split influence curves averaged into
# (9.14)-(9.15), and the separation diagnostics.

.TMLDTA_METHODS <- c("cv-tmle", "sample-split", "naive")
.tmldta_EPS <- 1e-9

#' .tmldta_logit
#'
#' A step of the tmldta_native implementation. Called by \code{.split_specific_tmle}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param p Coerced to numeric by the body, with \code{as.numeric}.
#' @return A numeric value.
#' @export
.tmldta_logit <- function(p) {
  q <- min(max(as.numeric(p), .tmldta_EPS), 1 - .tmldta_EPS)
  log(q / (1 - q))
}

#' .tmldta_expit
#'
#' A step of the tmldta_native implementation. Called by \code{.fit_g_dta}, \code{.fit_q_dta}, \code{.split_specific_tmle} and 1 others in the module.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x Numeric; combined arithmetically in the body.
#' @return One of two values, depending on the branch taken.
#' @export
.tmldta_expit <- function(x) {
  if (x > -700) 1 / (1 + exp(-x)) else 0
}

#' .tmldta_qnorm
#'
#' A step of the tmldta_native implementation. Called by \code{morie_tmle_data_adaptive}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param p Passed to \code{qnorm}.
#' @return The value of \code{qnorm}.
#' @export
.tmldta_qnorm <- function(p) {
  qnorm(p, 0, 1)
}

# Logistic IRLS that returns a coefficient vector for a 0/1 outcome.
#' Logistic IRLS that returns a coefficient vector for a 0/1 outcome
#'
#' A step of the tmldta_native implementation. Called by \code{.fit_g_dta}, \code{.fit_q_dta}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param Z A matrix; the body checks with \code{is.matrix}.
#' @param a A vector; its length is taken.
#' @param ridge Numeric; combined arithmetically in the body. Defaults to \code{1e-08}.
#' @param max_iter A count; the body uses it as \code{seq_len(...)}. Defaults to \code{50L}.
#' @param tol Passed to \code{<}. Defaults to \code{1e-10}.
#' @return The value of \code{b}, as built in the body.
#' @export
.tmldta_logit_irls <- function(Z, a, ridge = 1e-8, max_iter = 50L,
                        tol = 1e-10) {
  n <- length(a)
  if (is.matrix(Z)) {
    X <- Z
  } else {
    X <- do.call(rbind, Z)
  }
  p <- ncol(X)
  b <- rep(0, p)
  XtWX <- matrix(0, p, p)
  XtWz <- numeric(p)
  for (it in seq_len(max_iter)) {
    eta <- as.numeric(X %*% b)
    pc <- pmin(pmax(.tmldta_expit(eta), .tmldta_EPS), 1 - .tmldta_EPS)
    W <- pc * (1 - pc)
    z <- eta + (a - pc) / W
    XtWX <- crossprod(X, X * W) + ridge * diag(p)
    XtWz <- crossprod(X, W * z)
    b_new <- tryCatch(solve(XtWX, XtWz),
                      error = function(e) solve(XtWX + 1e-8 * diag(p), XtWz))
    if (max(abs(b_new - b)) < tol) { b <- b_new; break }
    b <- b_new
  }
  b
}

# Q surface for the data-adaptive target parameter.
#' Q surface for the data-adaptive target parameter
#'
#' A step of the tmldta_native implementation. Called by \code{.discover_levels}, \code{.split_specific_tmle}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param ys A vector; indexed elementwise.
#' @param A_ A vector; indexed elementwise.
#' @param W A matrix; indexed by row and column.
#' @param levels A vector; indexed elementwise.
#' @param rows Iterated over elementwise, with \code{lapply}.
#' @param ridge Numeric; passed to \code{max}.
#' @return A list with \code{q}, \code{b}.
#' @export
.fit_q_dta <- function(ys, A_, W, levels, rows, ridge) {
  ref <- levels[1]
  others <- levels[-1]
  p <- if (is.matrix(W)) ncol(W) else
    if (length(W) > 0L) ncol(W[[1L]]) else 0L
  rowf <- function(a, i) {
    d <- as.numeric(levels(others) == a) * 1.0
    if (is.matrix(W)) {
      r <- c(1, d, W[i, ])
      if (p > 0) {
        for (t in seq_along(others)) r <- c(r, d[t] * W[i, ])
      }
    } else {
      r <- c(1, d, W[[i]])
      if (p > 0) for (t in seq_along(others)) r <- c(r, d[t] * W[[i]])
    }
    r
  }
  X <- lapply(rows, function(i) rowf(A_[i], i))
  Xm <- do.call(rbind, X)
  av <- ys[rows]
  b <- .tmldta_logit_irls(Xm, av, ridge = max(ridge, 1e-10))
  qf <- function(a, i) {
    r <- rowf(a, i)
    .tmldta_expit(sum(r * b))
  }
  list(q = qf, b = b)
}

# Three-category treatment propensity, normalised.
#' Three-category treatment propensity, normalised
#'
#' A step of the tmldta_native implementation. Called by \code{.split_specific_tmle}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param A_ A vector; its length is taken.
#' @param W A vector; its length is taken.
#' @param aL Passed to \code{==}.
#' @param aH Passed to \code{==}.
#' @param rows See Usage.
#' @param ridge Numeric; passed to \code{max}.
#' @param trim Numeric; passed to \code{max}.
#' @return A list with \code{gH}, \code{gL}.
#' @export
.fit_g_dta <- function(A_, W, aL, aH, rows, ridge, trim) {
  n <- length(A_)
  if (is.matrix(W)) {
    X <- cbind(1, W)
  } else if (length(W) > 0L) {
    X <- do.call(rbind, lapply(W, function(r) c(1, r)))
  } else {
    X <- matrix(1, n, 1)
  }
  Xr <- X[rows, , drop = FALSE]
  catf <- function(mask) {
    b <- .tmldta_logit_irls(Xr, mask[rows], ridge = max(ridge, 1e-10))
    .tmldta_expit(as.numeric(X %*% b))
  }
  pH <- catf(as.numeric(A_ == aH))
  pL <- catf(as.numeric(A_ == aL))
  pO <- catf(as.numeric(!(A_ %in% c(aL, aH))))
  gH <- gL <- numeric(n)
  for (i in seq_len(n)) {
    tot <- pH[i] + pL[i] + pO[i]
    if (tot <= 0) { gH[i] <- 0.5; gL[i] <- 0.5; next }
    gH[i] <- min(max(pH[i] / tot, trim), 1 - trim)
    gL[i] <- min(max(pL[i] / tot, trim), 1 - trim)
  }
  list(gH = gH, gL = gL)
}

# Discover the data-adaptive levels (argmin and argmax of mean Q).
#' Discover the data-adaptive levels (argmin and argmax of mean Q)
#'
#' A step of the tmldta_native implementation. Called by \code{morie_tmle_data_adaptive}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param ys Passed to \code{.fit_q_dta}.
#' @param A_ Passed to \code{.fit_q_dta}.
#' @param W Passed to \code{.fit_q_dta}.
#' @param levels A vector; indexed elementwise.
#' @param rows Passed to \code{.fit_q_dta}.
#' @param eval_rows Iterated over elementwise, with \code{vapply}.
#' @param ridge Passed to \code{.fit_q_dta}.
#' @return A list with \code{aL}, \code{aH}, \code{info}.
#' @export
.discover_levels <- function(ys, A_, W, levels, rows, eval_rows, ridge) {
  fit <- .fit_q_dta(ys, A_, W, levels, rows, ridge)
  means <- sapply(levels, function(a)
    mean(vapply(eval_rows, function(i) fit$q(a, i), numeric(1))))
  names(means) <- as.character(levels)
  aL <- levels[which.min(means)]
  aH <- levels[which.max(means)]
  list(aL = aL, aH = aH, info = list(means = means,
                                      spread = max(means) - min(means)))
}

# Solve the per-split TMLE at fixed levels.
#' Solve the per-split TMLE at fixed levels
#'
#' A step of the tmldta_native implementation. Called by \code{morie_tmle_data_adaptive}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param ys A vector; its length is taken and its elements indexed.
#' @param A_ A vector; indexed elementwise.
#' @param W Passed to \code{.fit_q_dta}.
#' @param levels Passed to \code{.fit_q_dta}.
#' @param aL Passed to \code{.fit_g_dta}.
#' @param aH Passed to \code{.fit_g_dta}.
#' @param fit_rows Passed to \code{.fit_q_dta}.
#' @param est_rows A vector; its length is taken.
#' @param ridge Passed to \code{.fit_q_dta}.
#' @param trim Passed to \code{.fit_g_dta}.
#' @param target A flag; the body branches on it.
#' @return A list with \code{psi}, \code{D}, \code{info}.
#' @export
.split_specific_tmle <- function(ys, A_, W, levels, aL, aH,
                                  fit_rows, est_rows, ridge, trim,
                                  target) {
  n <- length(ys)
  fit <- .fit_q_dta(ys, A_, W, levels, fit_rows, ridge)
  g <- .fit_g_dta(A_, W, aL, aH, fit_rows, ridge, trim)
  H <- numeric(n)
  for (i in seq_len(n)) {
    H[i] <- (if (A_[i] == aH) 1 / g$gH[i] else 0) -
            (if (A_[i] == aL) 1 / g$gL[i] else 0)
  }
  off <- vapply(seq_len(n), function(i) .tmldta_logit(fit$q(A_[i], i)),
                numeric(1))
  eps <- 0
  if (target) {
    for (it in seq_len(100L)) {
      num <- den <- 0
      for (i in est_rows) {
        p <- .tmldta_expit(off[i] + eps * H[i])
        num <- num + H[i] * (ys[i] - p)
        den <- den + H[i] * H[i] * p * (1 - p)
      }
      if (den < 1e-12) break
      step <- num / den
      eps <- eps + step
      if (abs(step) < 1e-12) break
    }
  }
  qstar <- function(a, i) {
    h <- if (a == aH) 1 / g$gH[i] else -1 / g$gL[i]
    .tmldta_expit(.tmldta_logit(fit$q(a, i)) + eps * h)
  }
  m <- length(est_rows)
  psi <- mean(vapply(est_rows, function(i) qstar(aH, i) - qstar(aL, i),
                     numeric(1)))
  D <- vapply(est_rows, function(i) {
    resid <- .tmldta_expit(off[i] + eps * H[i])
    H[i] * (ys[i] - resid) + qstar(aH, i) - qstar(aL, i) - psi
  }, numeric(1))
  list(psi = psi, D = D, info = list(eps = eps,
                                       max_weight = max(vapply(
                                         est_rows, function(i)
                                           max(1 / g$gH[i], 1 / g$gL[i]),
                                           numeric(1)))))
}

#' .folds_dta
#'
#' A step of the tmldta_native implementation. Called by \code{morie_tmle_data_adaptive}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param n A count; the body uses it as \code{seq_len(...)}.
#' @param n_folds Coerced to integer by the body, with \code{as.integer}.
#' @return The value of \code{lapply}.
#' @export
.folds_dta <- function(n, n_folds) {
  V <- max(2, min(as.integer(n_folds), n))
  lapply(seq_len(V), function(v)
    which(seq_len(n) %% V == v - 1L))
}

#' Contrast between data-discovered exposure levels, done honestly
#'
#' @param y,D,X Outcome, exposure, covariates.
#' @param candidate_strata Optional vector of candidate exposure
#'   levels; defaults to the sorted distinct exposures.
#' @param method One of \code{"cv-tmle"}, \code{"sample-split"},
#'   \code{"naive"}.
#' @param n_folds Number of V-fold splits.
#' @param trim Positivity trim for the propensity score.
#' @param ridge Optional ridge for the nuisance fits.
#' @param level Confidence level for the interval.
#' @param bounds Optional \code{c(lower, upper)}; defaults to the
#'   data range.
#' @return A list with \code{estimate}, \code{se}, \code{ci}, the
#'   per-split levels and estimates, and the separation diagnostics.
#' @references Hubbard, A. E. et al. (2018).
#' @export
morie_tmle_data_adaptive <- function(y, D, X, candidate_strata = NULL,
                                     method = "cv-tmle", n_folds = 10,
                                     trim = 0.01, ridge = 1e-8,
                                     level = 0.95, bounds = NULL) {
  if (!(method %in% .TMLDTA_METHODS))
    stop("tmldta: method must be one of cv-tmle/sample-split/naive")
  yv <- as.numeric(y); Av <- as.numeric(D)
  n <- length(yv)
  if (length(Av) != n) stop("tmldta: outcome/exposure length mismatch")
  Wm <- if (is.null(X)) matrix(0, n, 0) else as.matrix(X)
  if (nrow(Wm) != n) stop("tmldta: covariate row count mismatch")
  if (!(trim > 0 && trim < 0.5))
    stop("tmldta: trim must be in (0, 0.5)")
  if (n < 8) stop("tmldta: need at least 8 observations")
  if (is.null(candidate_strata)) {
    lv <- sort(unique(Av))
  } else {
    lv <- unique(as.numeric(candidate_strata))
  }
  if (length(lv) < 2L) stop("tmldta: need at least 2 exposure levels")
  missing_ <- setdiff(lv, Av)
  if (length(missing_) > 0L)
    stop("tmldta: candidate levels never occur")
  if (is.null(bounds)) { lo <- min(yv); hi <- max(yv) }
  else { lo <- as.numeric(bounds[1]); hi <- as.numeric(bounds[2]) }
  rng <- hi - lo
  if (rng <= 0) stop("tmldta: the outcome has no range")
  if (any(yv < lo - 1e-12 | yv > hi + 1e-12))
    stop("tmldta: an outcome falls outside bounds")
  ys <- pmin(pmax((yv - lo) / rng, 0), 1)
  all_rows <- seq_len(n)
  Wl <- if (ncol(Wm) == 0L) list(rep(list(numeric(0)), n)) else
    lapply(seq_len(n), function(i) Wm[i, ])
  if (method == "naive") {
    dl <- .discover_levels(ys, Av, Wl, lv, all_rows, all_rows, ridge)
    sp <- .split_specific_tmle(ys, Av, Wl, lv, dl$aL, dl$aH, all_rows,
                                all_rows, ridge, trim, target = FALSE)
    splits <- list(list(aL = dl$aL, aH = dl$aH,
                         estimate = rng * sp$psi, n_est = n))
    sigma2 <- sum(sp$D^2) / n
    psi_hat <- sp$psi
    eps_all <- c(0)
  } else {
    folds <- .folds_dta(n, n_folds)
    splits <- list(); per_split <- list(); ics <- list(); eps_all <- c()
    for (est in folds) {
      gen <- setdiff(all_rows, est)
      if (length(gen) == 0L || length(est) == 0L) next
      dl <- .discover_levels(ys, Av, Wl, lv, gen, gen, ridge)
      fit_rows <- if (method == "cv-tmle") gen else est
      sp <- .split_specific_tmle(ys, Av, Wl, lv, dl$aL, dl$aH,
                                  fit_rows, est, ridge, trim,
                                  target = TRUE)
      per_split[[length(per_split) + 1L]] <- sp$psi
      eps_all <- c(eps_all, sp$info$eps)
      ics[[length(ics) + 1L]] <- sp$D
      splits[[length(splits) + 1L]] <- list(aL = dl$aL, aH = dl$aH,
                                             estimate = rng * sp$psi,
                                             n_est = length(est))
    }
    if (length(per_split) == 0L) stop("tmldta: no usable splits")
    psi_hat <- mean(unlist(per_split))
    sigma2 <- mean(vapply(ics, function(ic) sum(ic^2) / length(ic),
                          numeric(1)))
  }
  psi <- rng * psi_hat
  se <- rng * sqrt(sigma2 / n)
  z <- .tmldta_qnorm(0.5 + 0.5 * level)
  chosen <- table(unlist(lapply(splits,
                                 function(s) paste0(s$aL, ",", s$aH))))
  modal <- names(which.max(chosen))
  modal_pair <- as.numeric(strsplit(modal, ",")[[1]])
  agreement <- max(chosen) / length(splits)
  # separation: closest gap between the chosen levels' mean and a rival
  di2 <- .discover_levels(ys, Av, Wl, lv, all_rows, all_rows, ridge)
  ord <- sort(di2$info$means)
  sep <- min(ord[2] - ord[1], ord[length(ord)] - ord[length(ord) - 1])
  separation <- sep * rng
  list(estimate = psi, se = se, n = n,
       ci = c(psi - z * se, psi + z * se), level = level,
       levels_by_split = lapply(splits, function(s) c(s$aL, s$aH)),
       level_counts = as.list(chosen),
       modal_levels = modal_pair,
       level_agreement = agreement,
       separation = separation,
       near_tie = (separation < 2 * se) || (agreement < 0.6),
       level_means = di2$info$means * rng + lo,
       split_estimates = vapply(splits, function(s) s$estimate, numeric(1)),
       n_splits = length(splits), epsilon = eps_all,
       candidate_levels = lv, method = method,
       sigma = sqrt(sigma2) * rng,
       algorithm = paste("data-adaptive target parameter, Hubbard,",
                         "Kennedy & van der Laan (2018) Ch. 9",
                         "eq. (9.2)-(9.16)"))
}

#' Rank the columns of X by data-adaptive importance
#'
#' For each column the contrast is estimated with that column as the
#' exposure and everything else as the covariate, and the results are
#' sorted by absolute estimate.
#'
#' @param y Numeric outcome.
#' @param X Numeric matrix of predictors.
#' @param candidate_strata Optional levels to search over.
#' @param method One of the three honest methods.
#' @param n_folds Number of V-fold splits.
#' @param names Optional variable names; defaults to \code{"X1"},
#'   \code{"X2"}, ...
#' @param ... Forwarded to \code{morie_tmle_data_adaptive}.
#' @return A list of per-variable results sorted by absolute estimate.
#' @export
morie_variable_importance <- function(y, X, candidate_strata = NULL,
                                      method = "cv-tmle", n_folds = 10,
                                      names = NULL, ...) {
  Xm <- as.matrix(X); n <- nrow(Xm)
  p <- ncol(Xm)
  if (p < 2L) stop("variable_importance: need at least 2 columns")
  nm <- if (is.null(names)) paste0("X", seq_len(p)) else names
  if (length(nm) != p)
    stop("variable_importance: name/column count mismatch")
  out <- list()
  for (j in seq_len(p)) {
    A_ <- Xm[, j]
    W <- if (p > 1L) Xm[, -j, drop = FALSE] else matrix(0, n, 0)
    r <- morie_tmle_data_adaptive(y, A_, W,
                                  candidate_strata = candidate_strata,
                                  method = method, n_folds = n_folds, ...)
    out[[j]] <- list(variable = nm[j], index = j - 1L,
                      estimate = r$estimate, se = r$se,
                      ci = r$ci, levels = r$modal_levels)
  }
  ord <- order(-abs(vapply(out, function(d) d$estimate, numeric(1))))
  out <- out[ord]
  for (rank_ in seq_along(out)) out[[rank_]]$rank <- rank_
  out
}

#' Compact one-line summary of the tmldta recipe
#'
#' @return A character string.
#' @export
morie_tmldta_cheatsheet <- function() {
  paste("tmldta: levels found in the data (aL = argmin, aH = argmax",
        "of mean Q(a,W)) then the contrast estimated -- but NOT on",
        "the same rows. Naive reuse is structurally >= 0 under the",
        "null. cv-tmle fits Q and g on the parameter-generating split",
        "and only epsilon on the estimation split; average the split",
        "estimates (9.14), variance from the average of the split",
        "influence curves (9.15).")
}

morie_tmledataadaptive <- morie_tmle_data_adaptive

# house entry point: the package exports one morie_<module>
morie_tmldta <- morie_tmle_data_adaptive
