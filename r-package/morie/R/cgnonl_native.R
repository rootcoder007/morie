# Nonlinear conjugate gradients.
# Source: Fletcher, R., & Reeves, C. M. (1964) "Function minimization by
# conjugate gradients", The Computer Journal 7(2), 149-154,
# doi:10.1093/comjnl/7.2.149.
# Plus: Polak, E., & Ribiere, G. (1969) "Note sur la convergence de
# methodes de directions conjuguees", Revue francaise d'informatique
# et de recherche operationnelle, Serie rouge 3(R1), 35-43,
# http://www.numdam.org/article/M2AN_1969__3_1_35_0.pdf (eq. 3.20).
# Plus: Shewchuk, J. R. (1994) "An Introduction to the Conjugate
# Gradient Method Without the Agonizing Pain", CMU-CS-94-125, sec. 14.1
# (the max(beta, 0) safeguard).

.CGNONL_BETA_RULES <- c("fletcher-reeves", "polak-ribiere", "polak-ribiere-plus")
.CGNONL_SEARCHES <- c("fletcher-reeves", "exact-quadratic")

#' .cgnonl_dot
#'
#' A step of the cgnonl_native implementation. Called by
#' \code{.cgnonl_exact_quadratic_step}, \code{cgnonl_beta_fletcher_reeves},
#' \code{cgnonl_beta_polak_ribiere} and 2 others in the module.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param a Numeric; combined arithmetically in the body.
#' @param b Numeric; combined arithmetically in the body.
#' @return A numeric value.
#' @export
#' @examples
#' A <- matrix(c(4, 1, 0.5, 1, 3, 0.8, 0.5, 0.8, 2), nrow = 3)
#' b <- c(1.5, 2.5, 3.5)
#' res <- .cgnonl_dot(a = A, b = b)
#' res
.cgnonl_dot <- function(a, b) sum(a * b)

#' cgnonl_beta_fletcher_reeves
#'
#' A step of the cgnonl_native implementation. Called by \code{.cgnonl_beta}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param g_new Passed to \code{.cgnonl_dot}.
#' @param g_old Passed to \code{.cgnonl_dot}.
#' @return A numeric value.
#' @export
cgnonl_beta_fletcher_reeves <- function(g_new, g_old) {
  den <- .cgnonl_dot(g_old, g_old)
  if (den <= 0.0) return(0.0)
  .cgnonl_dot(g_new, g_new) / den
}

#' cgnonl_beta_polak_ribiere
#'
#' A step of the cgnonl_native implementation. Called by \code{.cgnonl_beta}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param g_new Numeric; combined arithmetically in the body.
#' @param g_old Numeric; combined arithmetically in the body.
#' @param plus A flag; the body branches on it. Defaults to \code{FALSE}.
#' @return One of two values, depending on the branch taken.
#' @export
cgnonl_beta_polak_ribiere <- function(g_new, g_old, plus = FALSE) {
  den <- .cgnonl_dot(g_old, g_old)
  if (den <= 0.0) return(0.0)
  diff <- g_new - g_old
  num <- .cgnonl_dot(g_new, diff)
  b <- num / den
  if (plus) max(b, 0.0) else b
}

#' .cgnonl_beta
#'
#' A step of the cgnonl_native implementation. Called by \code{cgnonl_nonlinear_cg}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param rule One of \code{"fletcher-reeves"}, \code{"polak-ribiere"}, \code{"polak-ribiere-plus"}.
#' @param g_new Passed to \code{cgnonl_beta_fletcher_reeves}.
#' @param g_old Passed to \code{cgnonl_beta_fletcher_reeves}.
#' @return Nothing; this branch always raises.
#' @export
.cgnonl_beta <- function(rule, g_new, g_old) {
  if (rule == "fletcher-reeves") return(cgnonl_beta_fletcher_reeves(g_new, g_old))
  if (rule == "polak-ribiere") return(cgnonl_beta_polak_ribiere(g_new, g_old, plus = FALSE))
  if (rule == "polak-ribiere-plus") return(cgnonl_beta_polak_ribiere(g_new, g_old, plus = TRUE))
  stop(sprintf("cgnonl: beta must be one of %s",
               paste(.CGNONL_BETA_RULES, collapse = ", ")))
}

#' cgnonl_cubic_interpolate
#'
#' A step of the cgnonl_native implementation. Called by \code{cgnonl_line_search_fr}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param ta Numeric; passed to \code{max}.
#' @param fa Numeric; combined arithmetically in the body.
#' @param da Numeric; combined arithmetically in the body.
#' @param tb Numeric; passed to \code{max}.
#' @param fb Numeric; combined arithmetically in the body.
#' @param db Numeric; combined arithmetically in the body.
#' @return The value of \code{t}, as built in the body.
#' @export
cgnonl_cubic_interpolate <- function(ta, fa, da, tb, fb, db) {
  h <- tb - ta
  if (h == 0.0) return(ta)
  z <- 3.0 * (fa - fb) / h + da + db
  disc <- z * z - da * db
  if (disc < 0.0) return(0.5 * (ta + tb))
  w <- sqrt(disc)
  denom <- db - da + 2.0 * w
  if (denom == 0.0) return(0.5 * (ta + tb))
  t <- tb - h * (db + w - z) / denom
  lo <- min(ta, tb)
  hi <- max(ta, tb)
  if (!(lo <= t && t <= hi)) return(0.5 * (ta + tb))
  t
}

#' cgnonl_line_search_fr
#'
#' A step of the cgnonl_native implementation. Called by \code{cgnonl_nonlinear_cg}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param f Accepted by the signature and not used anywhere in the body.
#' @param grad Accepted by the signature and not used anywhere in the body.
#' @param x A vector; its length is taken.
#' @param p Numeric; combined arithmetically in the body.
#' @param f0 Numeric; combined arithmetically in the body.
#' @param g0 Passed to \code{.cgnonl_dot}.
#' @param est Optional; may be \code{NULL}. Coerced to numeric by the body, with \code{as.numeric}.
#' @param max_double A count; the body uses it as \code{seq_len(...)}. Defaults to \code{60L}.
#' @param max_cubic A count; the body uses it as \code{seq_len(...)}. Defaults to \code{40L}.
#' @param tol Numeric; combined arithmetically in the body. Defaults to \code{1e-12}.
#' @return A list with \code{t}, \code{x}, \code{f}, \code{g}, \code{n_eval}.
#' @export
cgnonl_line_search_fr <- function(f, grad, x, p, f0, g0, est = NULL,
                                  max_double = 60L, max_cubic = 40L,
                                  tol = 1e-12) {
  n <- length(x)
  slope0 <- .cgnonl_dot(p, g0)
  if (slope0 >= 0.0) {
    stop(sprintf("cgnonl: the search direction is not a descent direction (p'g = %g >= 0)", slope0))
  }
  pnorm <- sqrt(.cgnonl_dot(p, p))
  if (pnorm <= 0.0) stop("cgnonl: the search direction is zero")
  unit <- 1.0 / pnorm
  if (is.null(est)) {
    h <- unit
  } else {
    k <- 2.0 * (as.numeric(est) - f0) / slope0
    h <- if (0.0 < k && k < unit) k else unit
  }
  psi <- function(t) {
    xt <- x + t * p
    list(xt = xt, ft = f(xt), gt = grad(xt))
  }
  evals <- 0L
  ta <- 0.0
  fa <- f0
  da <- slope0
  t <- h
  tb <- NULL
  fb <- NULL
  db <- NULL
  xb <- NULL
  gb <- NULL
  for (k in seq_len(max_double)) {
    r <- psi(t)
    evals <- evals + 1L
    dt <- .cgnonl_dot(p, r$gt)
    if (dt >= 0.0 || r$ft > fa) {
      tb <- t
      fb <- r$ft
      db <- dt
      xb <- r$xt
      gb <- r$gt
      break
    }
    ta <- t
    fa <- r$ft
    da <- dt
    t <- t * 2.0
  }
  if (is.null(tb)) {
    r <- psi(ta)
    return(list(t = ta, x = r$xt, f = r$ft, g = r$gt, n_eval = evals + 1L))
  }
  best_t <- tb
  best_x <- xb
  best_f <- fb
  best_g <- gb
  if (fa < fb) {
    r2 <- psi(ta)
    evals <- evals + 1L
    best_t <- ta
    best_x <- r2$xt
    best_f <- r2$ft
    best_g <- r2$gt
  }
  for (k in seq_len(max_cubic)) {
    if (abs(tb - ta) <= tol * max(1.0, abs(tb))) break
    tc <- cgnonl_cubic_interpolate(ta, fa, da, tb, fb, db)
    r3 <- psi(tc)
    evals <- evals + 1L
    dc <- .cgnonl_dot(p, r3$gt)
    if (r3$ft < best_f) {
      best_t <- tc
      best_x <- r3$xt
      best_f <- r3$ft
      best_g <- r3$gt
    }
    if (abs(dc) <= tol * max(1.0, abs(slope0))) {
      return(list(t = tc, x = r3$xt, f = r3$ft, g = r3$gt, n_eval = evals))
    }
    if (dc < 0.0) {
      ta <- tc
      fa <- r3$ft
      da <- dc
    } else {
      tb <- tc
      fb <- r3$ft
      db <- dc
    }
  }
  list(t = best_t, x = best_x, f = best_f, g = best_g, n_eval = evals)
}

#' .cgnonl_exact_quadratic_step
#'
#' A step of the cgnonl_native implementation. Called by \code{cgnonl_nonlinear_cg}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x Accepted by the signature and not used anywhere in the body.
#' @param p Passed to \code{.cgnonl_dot}.
#' @param g Passed to \code{.cgnonl_dot}.
#' @param hess_vec Accepted by the signature and not used anywhere in the body.
#' @return A numeric value.
#' @export
.cgnonl_exact_quadratic_step <- function(x, p, g, hess_vec) {
  ap <- hess_vec(p)
  den <- .cgnonl_dot(p, ap)
  if (den <= 0.0) {
    stop(sprintf("cgnonl: p'Ap = %g is not positive; the exact quadratic step needs a positive definite A", den))
  }
  -.cgnonl_dot(p, g) / den
}

#' cgnonl_nonlinear_cg
#'
#' A step of the cgnonl_native implementation. Called by \code{morie_cgnonl}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param f Passed to \code{cgnonl_line_search_fr}.
#' @param grad Passed to \code{cgnonl_line_search_fr}.
#' @param x0 Coerced to numeric by the body, with \code{as.numeric}.
#' @param beta Passed to \code{.cgnonl_beta}. Defaults to \code{"fletcher-reeves"}.
#' @param restart Optional; may be \code{NULL}. Numeric; combined arithmetically in the body.
#' @param max_iter Optional; may be \code{NULL}. Coerced to integer by the body, with
#' \code{as.integer}.
#' @param tol Numeric; combined arithmetically in the body. Defaults to \code{1e-10}.
#' @param est Passed to \code{cgnonl_line_search_fr}.
#' @param line_search Compared against \code{"exact-quadratic"}. Defaults to
#' \code{"fletcher-reeves"}.
#' @param hess_vec Optional; may be \code{NULL}. Passed to \code{.cgnonl_exact_quadratic_step}.
#' @param keep_path A flag; the body branches on it. Defaults to \code{FALSE}.
#' @return A list with \code{x}, \code{fun}, \code{grad}, \code{gnorm}, \code{n_iter},
#' \code{n_restart}, \code{n_feval}, \code{converged}, \code{betas}, \code{path},
#' \code{beta_rule}, \code{line_search}, \code{restart_every}, \code{method},
#' \code{note}.
#' @export
cgnonl_nonlinear_cg <- function(f, grad, x0, beta = "fletcher-reeves",
                                restart = NULL, max_iter = NULL,
                                tol = 1e-10, est = NULL,
                                line_search = "fletcher-reeves",
                                hess_vec = NULL, keep_path = FALSE) {
  if (!(beta %in% .CGNONL_BETA_RULES)) {
    stop(sprintf("cgnonl: beta must be one of %s",
                 paste(.CGNONL_BETA_RULES, collapse = ", ")))
  }
  if (!(line_search %in% .CGNONL_SEARCHES)) {
    stop(sprintf("cgnonl: line_search must be one of %s",
                 paste(.CGNONL_SEARCHES, collapse = ", ")))
  }
  if (line_search == "exact-quadratic" && is.null(hess_vec)) {
    stop("cgnonl: line_search='exact-quadratic' needs hess_vec, the map p -> Ap")
  }
  x <- as.numeric(x0)
  n <- length(x)
  if (n == 0L) stop("cgnonl: x0 is empty")
  if (is.null(restart)) restart <- n + 1L
  restart <- as.integer(restart)
  if (restart < 0L) stop("cgnonl: restart must not be negative")
  if (is.null(max_iter)) max_iter <- 200L * n
  if (as.integer(max_iter) < 1L) stop("cgnonl: max_iter must be at least 1")

  g <- as.numeric(grad(x))
  fx <- as.numeric(f(x))
  p <- -g
  evals <- 1L
  betas <- numeric(0)
  path <- if (isTRUE(keep_path)) list(as.numeric(x)) else list()
  restarts <- 0L
  it <- 0L
  converged <- .cgnonl_dot(g, g) <= tol * tol

  while (!converged && it < as.integer(max_iter)) {
    it <- it + 1L
    if (line_search == "exact-quadratic") {
      t <- .cgnonl_exact_quadratic_step(x, p, g, hess_vec)
      x_new <- x + t * p
      f_new <- as.numeric(f(x_new))
      g_new <- as.numeric(grad(x_new))
      evals <- evals + 1L
    } else {
      r <- cgnonl_line_search_fr(f, grad, x, p, fx, g, est = est)
      t <- r$t
      x_new <- r$x
      f_new <- r$f
      g_new <- r$g
      evals <- evals + r$n_eval
    }
    g_old <- g
    x <- x_new
    fx <- f_new
    g <- g_new
    if (isTRUE(keep_path)) path[[length(path) + 1L]] <- as.numeric(x)
    if (.cgnonl_dot(g, g) <= tol * tol) {
      converged <- TRUE
      break
    }
    if (restart != 0L && (it %% restart == 0L)) {
      p <- -g
      restarts <- restarts + 1L
      betas <- c(betas, 0.0)
    } else {
      b <- .cgnonl_beta(beta, g, g_old)
      betas <- c(betas, b)
      p <- -g + b * p
      if (.cgnonl_dot(p, g) >= 0.0) {
        p <- -g
        restarts <- restarts + 1L
      }
    }
  }

  list(
    x = x,
    fun = fx,
    grad = g,
    gnorm = sqrt(.cgnonl_dot(g, g)),
    n_iter = it,
    n_restart = restarts,
    n_feval = evals,
    converged = converged,
    betas = betas,
    path = path,
    beta_rule = beta,
    line_search = line_search,
    restart_every = restart,
    method = "Fletcher & Reeves (1964) eq. 20, nonlinear conjugate gradients",
    note = "storage is three vectors -- x, g and p -- which is the paper's stated advantage over Davidon-Fletcher-Powell; restarts to steepest descent every n+1 iterations, which preserves quadratic convergence because they are no more frequent than every n"
  )
}

#' morie_cgnonl
#'
#' A step of the cgnonl_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param f Passed to \code{cgnonl_nonlinear_cg}.
#' @param grad Passed to \code{cgnonl_nonlinear_cg}.
#' @param x0 Passed to \code{cgnonl_nonlinear_cg}.
#' @param beta Passed to \code{cgnonl_nonlinear_cg}. Defaults to \code{"fletcher-reeves"}.
#' @param restart Passed to \code{cgnonl_nonlinear_cg}.
#' @param max_iter Passed to \code{cgnonl_nonlinear_cg}.
#' @param tol Passed to \code{cgnonl_nonlinear_cg}. Defaults to \code{1e-10}.
#' @param est Passed to \code{cgnonl_nonlinear_cg}.
#' @param line_search Passed to \code{cgnonl_nonlinear_cg}. Defaults to \code{"fletcher-reeves"}.
#' @param hess_vec Passed to \code{cgnonl_nonlinear_cg}.
#' @param keep_path Passed to \code{cgnonl_nonlinear_cg}. Defaults to \code{FALSE}.
#' @return The value of \code{cgnonl_nonlinear_cg}.
#' @export
morie_cgnonl <- function(f, grad, x0, beta = "fletcher-reeves",
                         restart = NULL, max_iter = NULL,
                         tol = 1e-10, est = NULL,
                         line_search = "fletcher-reeves",
                         hess_vec = NULL, keep_path = FALSE) {
  cgnonl_nonlinear_cg(f, grad, x0, beta, restart, max_iter, tol, est,
                      line_search, hess_vec, keep_path)
}
