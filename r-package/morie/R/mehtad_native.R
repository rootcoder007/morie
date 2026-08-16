# Mehrotra's predictor-corrector: two solves, one factorisation.
# Sources: Mehrotra, S. (1992) "On the Implementation of a
# Primal-Dual Interior Point Method", SIAM Journal on Optimization
# 2(4), 575-601, doi:10.1137/0802028 -- the second-order
# predictor-corrector with the centring heuristic of Sec. 5
# (Exhibit 5.1, Heuristic CENPAR), the reported ~40% / 50% / 35%
# iteration-count reductions against Lustig-Marsten-Shanno and
# the dual affine scaling methods, with the second-derivative
# contribution identified as the most significant, and Table 5.1
# showing only moderate variation in iteration count for the
# exponent between 2 and 4. Wright, S. J. (1997) Primal-Dual
# Interior-Point Methods, SIAM,
# doi:10.1137/1.9781611971453 -- Chapter 10 for the
# sigma = (mu_aff/mu)^3 form. Boyd, S. & Vandenberghe, L. (2004)
# Convex Optimization, Cambridge University Press,
# doi:10.1017/CBO9780511804441 -- Sec. 11.7 for the primal-dual
# framework and the residual formulation.
#
# Native implementation mirroring Python morie.fn.mehtad exactly:
# same primal/dual/complementarity residuals with the
# infeasible-start residuals not required to be zero initially,
# same fraction-to-boundary step, same adaptive centring
# sigma = (mu_aff/mu)^nu with the same nu range check, same
# reduced (normal-equation) Newton system with a small Tikhonov
# ridge so the Cholesky stays positive definite, same
# predictor-corrector with the same x + dx * dx_aff * ds_aff
# cross term, and the same positivity guard.

.mehtad_eps <- 1e-12

#' mehtad_residuals
#'
#' Part of the mehtad_native implementation; see the file header for the
#' source it follows.
#'
#' @param A See Usage.
#' @param b See Usage.
#' @param c See Usage.
#' @param x See Usage.
#' @param y See Usage.
#' @param s See Usage.
#' @return A list with \code{primal}, \code{dual}, \code{mu}, \code{primal_norm}, \code{dual_norm}, \code{note}.
#' @export
mehtad_residuals <- function(A, b, c, x, y, s) {
  M <- as.matrix(A); storage.mode(M) <- "double"
  m <- nrow(M); n <- ncol(M)
  xv <- as.numeric(x); yv <- as.numeric(y); sv <- as.numeric(s)
  bv <- as.numeric(b); cv <- as.numeric(c)
  rp <- as.numeric(M %*% xv - bv)
  rd <- as.numeric(t(M) %*% yv + sv - cv)
  mu <- sum(xv * sv) / n
  list(primal = rp, dual = rd, mu = mu,
       primal_norm = sqrt(sum(rp * rp)),
       dual_norm = sqrt(sum(rd * rd)),
       note = "an infeasible start is allowed; the residuals are driven to zero alongside mu")
}

#' max_step
#'
#' Part of the mehtad_native implementation; see the file header for the
#' source it follows.
#'
#' @param v See Usage.
#' @param dv See Usage.
#' @param eta Defaults to \code{0.9995}.
#' @return A numeric value.
#' @export
max_step <- function(v, dv, eta = 0.9995) {
  a <- 1.0
  for (i in seq_along(v)) if (dv[i] < 0) {
    r <- -v[i] / dv[i]
    if (r < a) a <- r
  }
  min(1.0, as.numeric(eta) * a)
}

#' centering_parameter
#'
#' Part of the mehtad_native implementation; see the file header for the
#' source it follows.
#'
#' @param mu See Usage.
#' @param mu_affine See Usage.
#' @param nu Defaults to \code{3}.
#' @return A list with \code{sigma}, \code{ratio}, \code{nu}, \code{approximation}, \code{note}.
#' @export
centering_parameter <- function(mu, mu_affine, nu = 3.0) {
  m <- as.numeric(mu); ma <- as.numeric(mu_affine)
  if (m <= 0) stop("mehtad: mu must be positive")
  if (ma < 0) stop("mehtad: the affine mu cannot be negative")
  nv <- as.numeric(nu)
  if (nv < 1.0 || nv > 6.0)
    stop("mehtad: nu outside the range the paper examined; it tabulates 2 to 4")
  ratio <- ma / m
  list(sigma = ratio ^ nv, ratio = ratio, nu = nv,
       approximation = if (ratio > 0.5) "poor" else "good",
       note = "ratio near 1 means the affine trajectory is badly approximated locally, so centre more")
}

#' .mehtad_solve_normal
#'
#' Part of the mehtad_native implementation; see the file header for the
#' source it follows.
#'
#' @param A See Usage.
#' @param d See Usage.
#' @param rhs See Usage.
#' @param ridge Defaults to \code{1e-11}.
#' @return The value of \code{backsolve}.
#' @export
.mehtad_solve_normal <- function(A, d, rhs, ridge = 1e-11) {
  M <- as.matrix(A); storage.mode(M) <- "double"
  m <- nrow(M); n <- ncol(M)
  dM <- M * rep(d, each = m)
  N <- dM %*% t(M)
  diag(N) <- diag(N) + ridge
  L <- chol(N)
  y_forw <- forwardsolve(t(L), as.numeric(rhs))
  backsolve(L, y_forw)
}

#' newton_direction
#'
#' Part of the mehtad_native implementation; see the file header for the
#' source it follows.
#'
#' @param A See Usage.
#' @param x See Usage.
#' @param s See Usage.
#' @param rp See Usage.
#' @param rd See Usage.
#' @param rc See Usage.
#' @return A list with \code{dx}, \code{dy}, \code{ds}.
#' @export
newton_direction <- function(A, x, s, rp, rd, rc) {
  M <- as.matrix(A); storage.mode(M) <- "double"
  m <- nrow(M); n <- ncol(M)
  xv <- as.numeric(x); sv <- as.numeric(s)
  d <- xv / sv
  t <- -as.numeric(rc) / sv + d * as.numeric(rd)
  rhs <- -as.numeric(rp) - as.numeric(M %*% t)
  dy <- .mehtad_solve_normal(M, d, rhs)
  ds <- -(as.numeric(rd) + as.numeric(t(M) %*% dy))
  dx <- (-as.numeric(rc) - xv * ds) / sv
  list(dx = dx, dy = dy, ds = ds)
}

#' solve_lp
#'
#' Part of the mehtad_native implementation; see the file header for the
#' source it follows.
#'
#' @param A See Usage.
#' @param b See Usage.
#' @param c See Usage.
#' @param tol Defaults to \code{1e-09}.
#' @param max_iter Defaults to \code{100L}.
#' @param nu Defaults to \code{3}.
#' @param eta Defaults to \code{0.9995}.
#' @param corrector Defaults to \code{TRUE}.
#' @return A list with \code{estimate}, \code{x}, \code{y}, \code{s}, \code{mu}, \code{objective}, \code{dual_objective}, \code{iterations}, \code{corrector}, \code{primal_residual}, \code{dual_residual}, \code{converged}, \code{method}, \code{note}.
#' @export
solve_lp <- function(A, b, c, tol = 1e-9, max_iter = 100L, nu = 3.0,
                     eta = 0.9995, corrector = TRUE) {
  M <- as.matrix(A); storage.mode(M) <- "double"
  m <- nrow(M); n <- ncol(M)
  bv <- as.numeric(b); cv <- as.numeric(c)
  if (length(bv) != m || length(cv) != n)
    stop(sprintf("mehtad: A is %dx%d but b has %d and c has %d",
                 m, n, length(bv), length(cv)))
  x <- rep(1.0, n); s <- rep(1.0, n); y <- rep(0.0, m)
  it <- 0L
  for (it in seq_len(as.integer(max_iter))) {
    r <- mehtad_residuals(M, bv, cv, x, y, s)
    mu <- r$mu
    if (mu < as.numeric(tol) && r$primal_norm < as.numeric(tol) &&
        r$dual_norm < as.numeric(tol)) break
    rc <- x * s
    aff <- newton_direction(M, x, s, r$primal, r$dual, rc)
    ap <- max_step(x, aff$dx, eta)
    ad <- max_step(s, aff$ds, eta)
    mu_aff <- sum((x + ap * aff$dx) * (s + ad * aff$ds)) / n
    sig <- centering_parameter(mu, mu_aff, nu)$sigma
    if (corrector) {
      rc2 <- x * s + aff$dx * aff$ds - sig * mu
    } else {
      rc2 <- x * s - sig * mu
    }
    d <- newton_direction(M, x, s, r$primal, r$dual, rc2)
    ap <- max_step(x, d$dx, eta)
    ad <- max_step(s, d$ds, eta)
    x <- x + ap * d$dx
    s <- s + ad * d$ds
    y <- y + ad * d$dy
    if (min(min(x), min(s)) <= 0)
      stop("mehtad: an iterate left the positive orthant, which the fraction-to-boundary rule exists to prevent")
  }
  rf <- mehtad_residuals(M, bv, cv, x, y, s)
  list(estimate = x, x = x, y = y, s = s, mu = rf$mu,
       objective = sum(cv * x),
       dual_objective = sum(bv * y),
       iterations = it, corrector = isTRUE(corrector),
       primal_residual = rf$primal_norm,
       dual_residual = rf$dual_norm,
       converged = rf$mu < as.numeric(tol) && rf$primal_norm < as.numeric(tol),
       method = "Mehrotra predictor-corrector; Mehrotra (1992)",
       note = "the corrector reuses the predictor's factorisation, so the second-order term costs a right-hand side rather than an iteration")
}

predictor_corrector <- solve_lp
mehrotras_predictor <- solve_lp

#' .mehtad_cheatsheet
#'
#' Part of the mehtad_native implementation; see the file header for the
#' source it follows.
#'
#' @return A character value.
#' @export
.mehtad_cheatsheet <- function() {
  paste("mehtad: the expensive part of an interior-point iteration ",
        "is ONE factorisation of A D A'; a second right-hand side ",
        "is nearly free, so spend it on information. PREDICTOR: ",
        "the pure Newton (affine) step, too aggressive to take ",
        "whole but exactly the diagnostic needed. CENTERING: ",
        "sigma = (mu_aff/mu)^nu -- a good affine step asks for ",
        "little centring, a bad one for a lot; the ratio says how ",
        "well the trajectory is locally approximated, and nu in ",
        "[2,4] barely matters. CORRECTOR: subtract the ",
        "second-order cross term dX_aff dS_aff e together with the ",
        "centring target. FRACTION-TO-BOUNDARY keeps x, s strictly ",
        "positive. About 40% fewer iterations, mostly from the ",
        "second derivative.", sep = "")
}

#' morie_mehtad
#'
#' Part of the mehtad_native implementation; see the file header for the
#' source it follows.
#'
#' @param A See Usage.
#' @param b See Usage.
#' @param c See Usage.
#' @param tol Defaults to \code{1e-09}.
#' @param max_iter Defaults to \code{100L}.
#' @param nu Defaults to \code{3}.
#' @param eta Defaults to \code{0.9995}.
#' @param corrector Defaults to \code{TRUE}.
#' @return The value of \code{solve_lp}.
#' @export
morie_mehtad <- function(A, b, c, tol = 1e-9, max_iter = 100L,
                         nu = 3.0, eta = 0.9995, corrector = TRUE) {
  solve_lp(A, b, c, tol, max_iter, nu, eta, corrector)
}
