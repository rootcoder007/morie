# smoopt_native.R -- mirror of smoopt_python_reference.py

# Platt's SMO: the same dual, chosen differently.
# Sources: Platt, J. C. (1998) "Sequential Minimal Optimization: A
# Fast Algorithm for Training Support Vector Machines", Microsoft
# Research Technical Report MSR-TR-98-14. The decomposition to
# the smallest possible optimisation problem -- two Lagrange
# multipliers, because the linear equality constraint forces them
# to move together -- solved analytically so that no numerical QP
# optimisation is required; the outer loop alternating between
# single passes over the entire training set and repeated passes
# over the non-bound examples until all of them obey the KKT
# conditions within tolerance; the second-choice heuristic
# maximising |E_1 - E_2| as a proxy for the step size, with the
# fallback hierarchy over non-bound examples from a random
# position and then over all examples; and the computation of the
# threshold b from the resulting non-bound multipliers, taking
# the midpoint when both are at a bound. Chang, C.-C. & Lin, C.-J.
# (2011) "LIBSVM: A Library for Support Vector Machines", *ACM
# TIST* 2(3), Article 27, doi:10.1145/1961189.1961199. The
# maximal-violating-pair selection kept as the alternative route;
# implemented in :mod:`svmopt`. Cortes, C. & Vapnik, V. (1995)
# "Support-Vector Networks", *Machine Learning* 20(3), 273-297,
# doi:10.1007/BF00994018. The dual being solved.
#
# Native implementation mirroring Python morie.fn.smoopt exactly:
# the same two-multiplier analytic step, the same outer loop
# alternating full sweeps with non-bound sweeps, the same
# second-choice hierarchy over the non-bound set then the full
# set, the same threshold recomputation from non-bound
# multipliers with the midpoint when both are bound, and the
# same Platt sign convention f = sum a y K - b (negative of
# LIBSVM) so the threshold reported here is the negative of the
# threshold reported by svmopt on the same separator.

.SMOOPT_EPS <- 1e-12

# Vectorise the (n, n) gram matrix K into a column-stored numeric
# matrix if it isn't already.
.smoopt_K <- function(K) {
  if (is.matrix(K)) return(K)
  n <- length(K)
  M <- matrix(0, n, n)
  for (i in seq_len(n)) for (j in seq_len(n)) M[i, j] <- K[[i]][[j]]
  M
}

#' error_cache
#'
#' Part of the smoopt_native implementation; see the file header for the
#' source it follows.
#'
#' @param alpha See Usage.
#' @param y See Usage.
#' @param K See Usage.
#' @param b See Usage.
#' @return A numeric value.
#' @export
error_cache <- function(alpha, y, K, b) {
  a <- as.numeric(alpha)
  yy <- as.numeric(y)
  n <- length(a)
  Km <- .smoopt_K(K)
  f <- as.numeric(Km %*% (a * yy)) - as.numeric(b)
  f - yy
}

#' violates_kkt
#'
#' Part of the smoopt_native implementation; see the file header for the
#' source it follows.
#'
#' @param i See Usage.
#' @param alpha See Usage.
#' @param y See Usage.
#' @param E See Usage.
#' @param C See Usage.
#' @param tol Defaults to \code{0.001}.
#' @return A logical value.
#' @export
violates_kkt <- function(i, alpha, y, E, C, tol = 1e-3) {
  a <- as.numeric(alpha)[i]
  r <- as.numeric(y)[i] * as.numeric(E)[i]
  ((r < -as.numeric(tol)) && (a < as.numeric(C) - .SMOOPT_EPS)) ||
    ((r >  as.numeric(tol)) && (a > .SMOOPT_EPS))
}

#' outer_loop_schedule
#'
#' Part of the smoopt_native implementation; see the file header for the
#' source it follows.
#'
#' @param alpha See Usage.
#' @param C See Usage.
#' @param examine_all See Usage.
#' @return A list with \code{indices}, \code{kind}, \code{n_non_bound}, \code{note}.
#' @export
outer_loop_schedule <- function(alpha, C, examine_all) {
  a <- as.numeric(alpha)
  if (examine_all)
    return(list(indices = seq_along(a), kind = "all",
                note = "a full sweep catches violators at a bound"))
  nb <- which(a > .SMOOPT_EPS & a < as.numeric(C) - .SMOOPT_EPS)
  list(indices = nb, kind = "non_bound",
       n_non_bound = length(nb),
       note = "the non-bound set is where the action is")
}

#' second_choice
#'
#' Part of the smoopt_native implementation; see the file header for the
#' source it follows.
#'
#' @param i1 See Usage.
#' @param alpha See Usage.
#' @param y See Usage.
#' @param E See Usage.
#' @param C See Usage.
#' @param rng See Usage.
#' @param tol Defaults to \code{0.001}.
#' @return A list with \code{index}, \code{level}, \code{note}.
#' @export
second_choice <- function(i1, alpha, y, E, C, rng, tol = 1e-3) {
  a <- as.numeric(alpha)
  n <- length(a)
  i1 <- as.integer(i1)
  nb <- which(a > .SMOOPT_EPS & a < as.numeric(C) - .SMOOPT_EPS)
  nb <- nb[nb != i1]
  if (length(nb) > 1L) {
    gaps <- abs(E[i1] - E[nb])
    j <- nb[which.max(gaps)]
    return(list(index = j, level = 1L, gap = max(gaps),
                note = paste("the analytic step is proportional to ",
                              "|E1 - E2|, so this maximises progress",
                              sep = "")))
  }
  start <- as.integer(.ghc_unif(rng, 1L) * max(n, 1L)) %% max(n, 1L)
  if (length(nb) > 0L) {
    j <- nb[(start %% length(nb)) + 1L]
    return(list(index = j, level = 2L,
                note = "non-bound examples from a random position"))
  }
  for (t in seq_len(n)) {
    j <- ((start + t - 1L) %% n) + 1L
    if (j != i1)
      return(list(index = j, level = 3L,
                  note = "all examples from a random position"))
  }
  list(index = NA_integer_, level = 4L,
       note = "no second index available; abandon this i1")
}

#' compute_threshold
#'
#' Part of the smoopt_native implementation; see the file header for the
#' source it follows.
#'
#' @param i1 See Usage.
#' @param i2 See Usage.
#' @param a1_new See Usage.
#' @param a2_new See Usage.
#' @param alpha See Usage.
#' @param y See Usage.
#' @param E See Usage.
#' @param K See Usage.
#' @param b See Usage.
#' @param C See Usage.
#' @return A list with \code{b}, \code{from}, \code{b1}, \code{b2}, \code{note}.
#' @export
compute_threshold <- function(i1, i2, a1_new, a2_new, alpha, y, E, K,
                              b, C) {
  yy <- as.numeric(y)
  a <- as.numeric(alpha)
  i <- as.integer(i1); j <- as.integer(i2)
  d1 <- as.numeric(a1_new) - a[i]
  d2 <- as.numeric(a2_new) - a[j]
  Km <- .smoopt_K(K)
  b1 <- as.numeric(b) + E[i] + yy[i] * d1 * Km[i, i] +
        yy[j] * d2 * Km[i, j]
  b2 <- as.numeric(b) + E[j] + yy[i] * d1 * Km[i, j] +
        yy[j] * d2 * Km[j, j]
  free1 <- as.numeric(a1_new) > .SMOOPT_EPS &&
           as.numeric(a1_new) < as.numeric(C) - .SMOOPT_EPS
  free2 <- as.numeric(a2_new) > .SMOOPT_EPS &&
           as.numeric(a2_new) < as.numeric(C) - .SMOOPT_EPS
  if (free1) return(list(b = b1, from = "i1", b1 = b1, b2 = b2))
  if (free2) return(list(b = b2, from = "i2", b1 = b1, b2 = b2))
  list(b = 0.5 * (b1 + b2), from = "midpoint", b1 = b1, b2 = b2,
       note = paste("both at a bound, so any value between b1 and ",
                    "b2 satisfies KKT", sep = ""))
}

# Analytic step bounds L <= a2_new <= H.  Mirrors
# svmopt._bounds exactly.
.smoopt_bounds <- function(i1, i2, alpha, y, C) {
  yy <- as.numeric(y); a <- as.numeric(alpha); C <- as.numeric(C)
  if (yy[i1] == yy[i2]) {
    L <- max(0, a[i1] + a[i2] - C)
    H <- min(C, a[i1] + a[i2])
  } else {
    L <- max(0, a[i2] - a[i1])
    H <- min(C, C + a[i2] - a[i1])
  }
  c(L, H)
}

# Dual objective of the C-SVM, in the Platt sign convention so it
# is directly comparable to svmopt.dual_objective up to the sign of
# the threshold term -- the threshold does not enter the dual
# itself.
.smoopt_dual_objective <- function(alpha, y, K) {
  a <- as.numeric(alpha); yy <- as.numeric(y); n <- length(a)
  Km <- .smoopt_K(K)
  sum(a) - 0.5 * as.numeric(t(a * yy) %*% Km %*% (a * yy))
}

#' smo_platt
#'
#' Part of the smoopt_native implementation; see the file header for the
#' source it follows.
#'
#' @param y See Usage.
#' @param K See Usage.
#' @param C Defaults to \code{1}.
#' @param tol Defaults to \code{0.001}.
#' @param eps Defaults to \code{1e-05}.
#' @param max_passes Defaults to \code{200L}.
#' @param seed Defaults to \code{0L}.
#' @return A list with \code{estimate}, \code{alpha}, \code{b}, \code{passes}, \code{full_passes}, \code{non_bound_passes}, \code{steps}, \code{support_vectors}, \code{n_sv}, \code{equality_residual}, \code{kkt_violations}, \code{objective}, \code{method}, \code{note}.
#' @export
smo_platt <- function(y, K, C = 1.0, tol = 1e-3, eps = 1e-5,
                      max_passes = 200L, seed = 0L) {
  yy <- as.numeric(y)
  n  <- length(yy)
  if (any(yy != -1 & yy != 1))
    stop("smoopt: labels must be -1 or +1")
  if (as.numeric(C) <= 0)
    stop("smoopt: C must be positive")
  rng <- .ghc_rng(as.numeric(seed))
  a <- rep(0, n)
  b <- 0
  examine_all <- TRUE
  passes <- 0L
  changed_total <- 0L
  full_passes <- 0L
  nb_passes   <- 0L
  Km <- .smoopt_K(K)
  while (passes < as.integer(max_passes)) {
    passes <- passes + 1L
    E <- error_cache(a, yy, Km, b)
    sched <- outer_loop_schedule(a, C, examine_all)
    if (sched$kind == "all") full_passes <- full_passes + 1L
    else                    nb_passes   <- nb_passes   + 1L
    changed <- 0L
    for (i1 in sched$indices) {
      if (!violates_kkt(i1, a, yy, E, C, tol)) next
      pick <- second_choice(i1, a, yy, E, C, rng, tol)
      i2 <- pick$index
      if (is.na(i2)) next
      LH <- .smoopt_bounds(i1, i2, a, yy, C)
      L <- LH[1]; H <- LH[2]
      if (H <= L + .SMOOPT_EPS) next
      eta <- Km[i1, i1] + Km[i2, i2] - 2 * Km[i1, i2]
      if (eta <= .SMOOPT_EPS) next
      a2_new <- a[i2] + yy[i2] * (E[i1] - E[i2]) / eta
      a2_new <- min(max(a2_new, L), H)
      if (abs(a2_new - a[i2]) < as.numeric(eps) *
          (a2_new + a[i2] + as.numeric(eps))) next
      a1_new <- a[i1] - yy[i1] * yy[i2] * (a2_new - a[i2])
      th <- compute_threshold(i1, i2, a1_new, a2_new, a, yy, E, Km, b, C)
      a[i1] <- a1_new; a[i2] <- a2_new
      b <- th$b
      E <- error_cache(a, yy, Km, b)
      changed <- changed + 1L
    }
    changed_total <- changed_total + changed
    if (examine_all) {
      examine_all <- FALSE
    } else if (changed == 0L) {
      examine_all <- TRUE
      if (passes > 1L) {
        E <- error_cache(a, yy, Km, b)
        if (!any(vapply(seq_len(n),
                        function(i) violates_kkt(i, a, yy, E, C, tol),
                        logical(1L))))
          break
      }
    }
  }
  E <- error_cache(a, yy, Km, b)
  sv <- which(a > .SMOOPT_EPS)
  list(estimate = a, alpha = a, b = b,
       passes = passes, full_passes = full_passes,
       non_bound_passes = nb_passes,
       steps = changed_total, support_vectors = sv,
       n_sv = length(sv),
       equality_residual = sum(a * yy),
       kkt_violations = sum(vapply(seq_len(n),
                                   function(i)
                                     violates_kkt(i, a, yy, E, C, tol),
                                   logical(1L))),
       objective = .smoopt_dual_objective(a, yy, Km),
       method = "SMO with Platt's heuristics; Platt (1998)",
       note = paste("same dual as svmopt, different working-set ",
                    "rule -- Platt's needs only the non-bound errors; ",
                    "note b follows Platt's f = sum(a y K) - b, the ",
                    "NEGATIVE of the LIBSVM convention used in svmopt",
                    sep = ""))
}

.smoopt_cheatsheet <- function() {
  paste("smoopt: same SVM dual as svmopt, different CHOICE of ",
        "pair. Two multipliers because the equality constraint ",
        "forces them to move together, and at two the QP is ",
        "analytic -- SMO calls NO inner QP solver. Outer loop ",
        "ALTERNATES: one full sweep, then repeated sweeps over the ",
        "NON-BOUND examples until they all satisfy KKT, then a ",
        "full sweep again -- bound examples rarely move, but ",
        "skipping them forever hides a violator sitting at a ",
        "bound. Inner heuristic maximises |E1 - E2|, since the ",
        "analytic step is proportional to it, with a fallback ",
        "hierarchy. b is RECOMPUTED each step, not accumulated.",
        sep = "")
}

sequential_minimal_optimization <- smo_platt
smo_solver <- smo_platt
smosolver <- smo_platt

# Native entry point.
morie_smoopt <- smo_platt
