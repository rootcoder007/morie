# Proximal gradient and FISTA for composite problems.
# Sources: Beck, A., & Teboulle, M. (2009) "A Fast Iterative
# Shrinkage-Thresholding Algorithm for Linear Inverse Problems",
# SIAM J. Imaging Sciences 2(1), 183-202. Sec. 2 (ISTA), eq. 4.1-4.3
# (FISTA), Sec. 4 (backtracking line search).
#
# Native implementation mirroring morie.fn.prxgms exactly: the same
# ISTA / FISTA update with the extrapolation (t-1)/t_next, the same
# monotone ISTA vs non-monotone FISTA choice, the same backtracking
# line search, and the same prox_1/L: the soft-threshold for the
# lasso thresholds at lam * t, not at t -- the lam must be carried
# through the iteration or the algorithm silently solves a different
# problem for every lam != 1.

#' Soft-thresholding
#'
#' Prox of \code{tau * |.|_1}, applied elementwise.
#'
#' @param v Numeric vector.
#' @param tau Non-negative threshold.
#' @return Numeric vector of the same length.
#' @export
soft_threshold <- function(v, tau) {
  v <- as.numeric(v)
  out <- numeric(length(v))
  for (i in seq_along(v)) {
    x <- v[i]
    out[i] <- if (x > tau) x - tau
              else if (x < -tau) x + tau
              else 0.0
  }
  out
}

#' Proximal gradient / FISTA on \code{f + g}
#'
#' \code{accelerate=TRUE} is FISTA (Beck & Teboulle 2009 eq. 4.1-4.3),
#' \code{FALSE} is plain ISTA (Sec. 2). \code{backtrack=TRUE} uses the
#' paper's Sec. 4 backtracking line search on L.
#'
#' @param fun Function \code{(x) -> float}, the smooth part \code{f}.
#' @param grad Function \code{(x) -> numeric}, the gradient of \code{f}.
#' @param prox Function \code{(x, t) -> numeric}, prox of \code{g * t}.
#' @param x0 Initial point.
#' @param L Initial Lipschitz estimate.
#' @param max_iter Integer, maximum iterations.
#' @param tol Stop when the step is at most \code{tol}.
#' @param accelerate If \code{TRUE} run FISTA, otherwise ISTA.
#' @param backtrack If \code{TRUE} use the backtracking line search.
#' @param eta Backtracking expansion factor.
#' @param g_fun Optional function \code{(x) -> float}, value of \code{g}.
#' @return A list with \code{estimate}, \code{x}, \code{fun},
#'   \code{objective}, \code{iterations}, \code{L}, \code{accelerated},
#'   \code{converged}, \code{method}.
#' @export
morie_prxgms <- function(fun, grad, prox, x0, L = 1.0, max_iter = 500L,
                         tol = 1e-10, accelerate = TRUE, backtrack = FALSE,
                         eta = 2.0, g_fun = NULL) {
  x <- as.numeric(x0)
  n <- length(x)
  L <- as.numeric(L)
  if (L <= 0)
    stop(sprintf("prox_gradient: L must be positive, got %r", L))
  y <- x
  t <- 1.0
  prev <- x
  obj <- numeric(0)
  it <- 0L
  converged <- FALSE
  Lk <- L
  for (it in seq_len(as.integer(max_iter))) {
    gy <- as.numeric(grad(y))
    if (backtrack) {
      fy <- as.numeric(fun(y))
      Lk <- L
      for (bt in seq_len(60L)) {
        z <- prox(y - gy / Lk, 1.0 / Lk)
        d <- z - y
        q <- fy + sum(gy * d) + 0.5 * Lk * sum(d^2)
        if (as.numeric(fun(z)) <= q + 1e-15) break
        Lk <- Lk * eta
      }
      L <- Lk
    } else {
      z <- prox(y - gy / Lk, 1.0 / Lk)
    }
    if (accelerate) {
      t_next <- 0.5 * (1.0 + sqrt(1.0 + 4.0 * t * t))
      w <- (t - 1.0) / t_next
      y <- z + w * (z - prev)
      t <- t_next
    } else {
      y <- z
    }
    step <- sqrt(sum((z - prev)^2))
    prev <- z
    fz <- as.numeric(fun(z))
    obj <- c(obj, fz + if (!is.null(g_fun)) as.numeric(g_fun(z)) else 0.0)
    if (step <= tol) {
      converged <- TRUE
      break
    }
  }
  list(estimate = prev, x = prev, fun = as.numeric(fun(prev)),
       objective = obj, iterations = as.integer(it),
       L = as.numeric(L), accelerated = isTRUE(accelerate),
       converged = converged,
       method = if (accelerate)
         "FISTA (Beck & Teboulle 2009, eq. 4.1-4.3)"
         else "ISTA (Beck & Teboulle 2009, Sec. 2)")
}

#' Lasso via ISTA / FISTA
#'
#' Minimises \code{1/2 ||Ax - b||^2 + lam * ||x||_1}; the prox is
#' soft-thresholding at \code{lam * t}, not at \code{t}.
#'
#' @param A Numeric matrix.
#' @param b Numeric vector.
#' @param lam Non-negative penalty.
#' @param max_iter Integer, maximum iterations.
#' @param tol Stop when the step is at most \code{tol}.
#' @param accelerate If \code{TRUE} run FISTA, otherwise ISTA.
#' @return A list with \code{estimate}, \code{x}, \code{fun},
#'   \code{objective}, \code{iterations}, \code{L}, \code{accelerated},
#'   \code{converged}, \code{method}, \code{lambda}, \code{L}.
#' @export
lasso_fista <- function(A, b, lam, max_iter = 500L, tol = 1e-10,
                        accelerate = TRUE) {
  Am <- as.matrix(A)
  storage.mode(Am) <- "double"
  bv <- as.numeric(b)
  n_rows <- nrow(Am)
  p <- ncol(Am)
  lam <- as.numeric(lam)
  f <- function(x) {
    r <- as.numeric(Am %*% x) - bv
    0.5 * sum(r^2)
  }
  g <- function(x) {
    r <- as.numeric(Am %*% x) - bv
    as.numeric(crossprod(Am, r))
  }
  # Power iteration for the largest eigenvalue of A'A
  v <- rep(1.0, p)
  L <- 1.0
  for (it in seq_len(200L)) {
    Av <- as.numeric(Am %*% v)
    u <- as.numeric(crossprod(Am, Av))
    nrm <- sqrt(sum(u^2))
    if (nrm <= 0) break
    v <- u / nrm
    L <- nrm
  }
  L <- max(L, 1e-12)
  prox <- function(v, t) soft_threshold(v, lam * t)
  res <- morie_prxgms(f, g, prox, rep(0.0, p), L = L,
                      max_iter = max_iter, tol = tol,
                      accelerate = accelerate,
                      g_fun = function(x) lam * sum(abs(x)))
  res$lambda <- lam
  res$L <- L
  res
}

#' @export
prox_gradient <- morie_prxgms

#' @export
prxgms <- morie_prxgms

#' @export
proximal_gradient_method <- morie_prxgms

#' @export
prxgms_cheatsheet <- function() {
  paste0("prxgms: ISTA/FISTA, x = prox_{g/L}(x - grad f / L), ",
         "t_{k+1} = (1+sqrt(1+4t^2))/2, ",
         "y = x + (t-1)/t_next (x - x_prev); ",
         "soft threshold for the lasso.")
}
