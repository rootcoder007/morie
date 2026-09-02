# VAR prewhitened kernel HAC covariance estimator.
# Sources: Andrews, D. W. K. & Monahan, J. C. (1992) "An Improved
# Heteroskedasticity and Autocorrelation Consistent Covariance Matrix
# Estimator", Econometrica 60(4), 953-966 -- eq. 2.2 (the VAR sponge
# for temporal dependence), eq. 2.3 (the kernel estimator on the
# residuals with the T/(T-l) correction), eq. 2.4 (the recolouring
# with D = (I - sum A_r)^{-1}), and the SVD cap of footnote 4 that
# keeps every eigenvalue of I - sum A_r at least 1 - cap from zero.
# Andrews, D. W. K. (1991) "Heteroskedasticity and Autocorrelation
# Consistent Covariance Matrix Estimation", Econometrica 59(3),
# 817-858 -- eq. 6.1 (the automatic bandwidth with the AR(1) plug-in)
# and eq. 6.4 (alpha(q) from p univariate AR(1) fits). Bartlett (1950),
# Parzen (1957), Newey & West (1987), Blackman & Tukey (1958) are the
# kernel references; their (q, k_q, int k^2) constants are recomputed
# from the kernel functions rather than transcribed.
#
# Native implementation mirroring morie.fn.andmnh exactly. No random
# draws are made; the function is fully deterministic.

.EIGENVALUE_CAP <- 0.97

#' Bartlett (triangular) kernel
#' @param x See Usage.
#' @export
bartlett_kernel <- function(x) {
  ax <- abs(as.numeric(x))
  if (ax <= 1) 1 - ax else 0
}

#' Parzen kernel
#' @param x See Usage.
#' @export
parzen_kernel <- function(x) {
  ax <- abs(as.numeric(x))
  if (ax <= 0.5) return(1 - 6 * ax * ax + 6 * ax ^ 3)
  if (ax <= 1) return(2 * (1 - ax) ^ 3)
  0
}

#' Quadratic spectral kernel
#' @param x See Usage.
#' @export
quadratic_spectral_kernel <- function(x) {
  x <- as.numeric(x)
  if (x == 0) return(1)
  z <- 6 * pi * x / 5
  25 / (12 * pi ^ 2 * x * x) * (sin(z) / z - cos(z))
}

#' Tukey-Hanning kernel
#' @param x See Usage.
#' @export
tukey_hanning_kernel <- function(x) {
  ax <- abs(as.numeric(x))
  if (ax <= 1) 0.5 * (1 + cos(pi * ax)) else 0
}

.MORIE_KERNELS <- list(bartlett = bartlett_kernel,
                       parzen = parzen_kernel,
                       qs = quadratic_spectral_kernel,
                       "tukey-hanning" = tukey_hanning_kernel)
.MORIE_KERNEL_CONSTANTS <- list(
  bartlett = c(1, 1.0, 2 / 3, 1),
  parzen = c(2, 6.0, 0.539285, 1),
  qs = c(2, 1.421223, 1.0, 0),
  "tukey-hanning" = c(2, pi ^ 2 / 4, 0.75, 1)
)

#' .morie_check_kernel
#'
#' A step of the andmnh_native implementation. Called by \code{automatic_bandwidth}, \code{kernel_hac}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param kernel Passed to \code{\%in\%}.
#' @return A list with \code{fn}, \code{const}.
#' @export
.morie_check_kernel <- function(kernel) {
  if (!(kernel %in% names(.MORIE_KERNELS)))
    stop("andmnh: kernel must be one of ",
         paste(names(.MORIE_KERNELS), collapse = ", "))
  list(fn = .MORIE_KERNELS[[kernel]],
       const = .MORIE_KERNEL_CONSTANTS[[kernel]])
}

#' Section 3 moment vectors
#' @param e See Usage.
#' @param X See Usage.
#' @export
moment_vectors <- function(e, X) {
  e <- as.numeric(e)
  rows <- as.matrix(X); storage.mode(rows) <- "double"
  if (length(e) != nrow(rows))
    stop("andmnh: length(e) must match nrow(X)")
  if (nrow(rows) == 0L) stop("andmnh: no observations")
  p <- ncol(rows)
  out <- matrix(0, nrow(rows), p)
  for (t in seq_len(nrow(rows)))
    for (j in seq_len(p)) out[t, j] <- rows[t, j] * e[t]
  out
}

#' .morie_svd
#'
#' A step of the andmnh_native implementation. Called by \code{prewhiten_var}, \code{singular_value_adjust}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param a A matrix; passed to \code{nrow}.
#' @return A list with \code{u}, \code{s}, \code{vt}.
#' @export
.morie_svd <- function(a) {
  a <- as.matrix(a)
  m <- nrow(a); n <- ncol(a); k <- min(m, n)
  AtA <- crossprod(a)
  ev <- eigen(AtA, symmetric = TRUE)
  s <- sqrt(pmax(ev$values, 0))
  # sort descending
  ord <- order(s, decreasing = TRUE)[seq_len(k)]
  s <- s[ord]
  V <- ev$vectors[, ord, drop = FALSE]
  # U = A V S^{-1}
  invs <- ifelse(s > 0, 1 / s, 0)
  U <- if (m >= n) a %*% (V * rep(invs, each = n)) else matrix(0, m, k)
  list(u = U, s = s, vt = t(V))
}

#' SVD cap on the prewhitening matrix
#' @param a See Usage.
#' @param cap See Usage.
#' @export
singular_value_adjust <- function(a, cap = .EIGENVALUE_CAP) {
  cap <- as.numeric(cap)
  if (!(cap > 0 && cap < 1))
    stop("andmnh: cap must lie strictly between 0 and 1")
  s <- .morie_svd(as.matrix(a))
  s2 <- pmin(s$s, cap)
  s$u %*% diag(s2, nrow = length(s2)) %*% s$vt
}

#' VAR prewhitening, equation 2.2
#' @param v See Usage.
#' @param order See Usage.
#' @param cap See Usage.
#' @param adjust See Usage.
#' @export
prewhiten_var <- function(v, order = 1L, cap = .EIGENVALUE_CAP,
                          adjust = TRUE) {
  rows <- as.matrix(v); storage.mode(rows) <- "double"
  n <- nrow(rows)
  if (n == 0L) stop("andmnh: no observations")
  p <- ncol(rows)
  order <- as.integer(order)
  if (order < 0L) stop("andmnh: VAR order must be non-negative")
  if (order == 0L) return(list(a_list = list(), resid = rows, D = diag(p)))
  if (n <= order * p + 1L)
    stop("andmnh: not enough observations for the requested VAR order")
  y <- rows[(order + 1L):n, , drop = FALSE]
  z <- matrix(0, n - order, order * p)
  for (t in (order + 1L):n) {
    for (r in seq_len(order)) {
      z[t - order, ((r - 1L) * p + 1L):(r * p)] <- rows[t - r, ]
    }
  }
  # lstsq via QR for stability
  qr_z <- qr(z); Q <- qr.Q(qr_z); R <- qr.R(qr_z)
  coef <- backsolve(R, crossprod(Q, y))
  a_list <- vector("list", order)
  for (r in seq_len(order)) {
    block <- matrix(0, p, p)
    for (i in seq_len(p)) for (j in seq_len(p))
      block[i, j] <- coef[(r - 1L) * p + j, i]
    a_list[[r]] <- block
  }
  if (adjust) {
    a_list <- lapply(a_list, function(a) singular_value_adjust(a, cap))
    if (order > 1L) {
      # For b > 1 capping each A_r is not enough. Shrink together so
      # the spectral norm of sum A_r is at most cap.
      for (it in seq_len(200L)) {
        tot <- a_list[[1]]
        for (k in 2:order) tot <- tot + a_list[[k]]
        smax <- max(.morie_svd(tot)$s)
        if (smax <= cap) break
        a_list <- lapply(a_list, function(a) a * (cap / smax))
      }
    }
  }
  resid <- matrix(0, n - order, p)
  for (t in (order + 1L):n) {
    pred <- rep(0, p)
    for (r in seq_len(order)) {
      ar <- a_list[[r]]
      for (i in seq_len(p)) for (j in seq_len(p))
        pred[i] <- pred[i] + ar[i, j] * rows[t - r, j]
    }
    resid[t - order, ] <- rows[t, ] - pred
  }
  tot <- a_list[[1]]
  for (k in 2:order) tot <- tot + a_list[[k]]
  D <- solve(diag(p) - tot)
  list(a_list = a_list, resid = resid, D = D)
}

#' AR(1) by least squares (no intercept)
#' @param x See Usage.
#' @export
ar1_fit <- function(x) {
  x <- as.numeric(x)
  n <- length(x)
  if (n < 3L) stop("andmnh: an AR(1) needs at least 3 observations")
  num <- sum(x[2:n] * x[1:(n - 1)])
  den <- sum(x[1:(n - 1)] ^ 2)
  rho <- if (den > 0) num / den else 0
  s2 <- sum((x[2:n] - rho * x[1:(n - 1)]) ^ 2) / (n - 1L)
  c(rho = rho, sigma2 = s2)
}

#' Andrews (1991) eq. 6.4, alpha(q) from p AR(1) fits
#' @param v See Usage.
#' @param q See Usage.
#' @param weights See Usage.
#' @export
alpha_ar1 <- function(v, q = 2L, weights = NULL) {
  rows <- as.matrix(v); storage.mode(rows) <- "double"
  if (nrow(rows) == 0L) stop("andmnh: no observations")
  p <- ncol(rows)
  if (is.null(weights)) w <- rep(1, p)
  else if (identical(weights, "drop_first")) w <- c(0, rep(1, p - 1))
  else {
    w <- as.numeric(weights)
    if (length(w) != p) stop("andmnh: weight length mismatch")
  }
  if (any(w < 0) || sum(w) <= 0)
    stop("andmnh: weights must be non-negative and not all zero")
  q <- as.integer(q)
  if (!(q %in% c(1L, 2L)))
    stop("andmnh: alpha(q) is given for q = 1 or 2")
  num <- 0; den <- 0
  fits <- list()
  for (a in seq_len(p)) {
    fit <- ar1_fit(rows[, a])
    fits[[a]] <- list(rho = unname(fit["rho"]), sigma2 = unname(fit["sigma2"]))
    rho <- fit["rho"]; s2 <- fit["sigma2"]
    s4 <- s2 * s2
    if (w[a] == 0) next
    if (q == 2L) {
      num <- num + w[a] * 4 * rho * rho * s4 / (1 - rho) ^ 8
    } else {
      num <- num + w[a] * 4 * rho * rho * s4 /
        ((1 - rho) ^ 6 * (1 + rho) ^ 2)
    }
    den <- den + w[a] * s4 / (1 - rho) ^ 4
  }
  if (den <= 0) stop("andmnh: the alpha(q) denominator vanished")
  list(alpha = num / den, fits = fits)
}

#' Andrews (1991) eq. 6.1, automatic plug-in bandwidth
#' @param v See Usage.
#' @param kernel See Usage.
#' @param weights See Usage.
#' @param n See Usage.
#' @export
automatic_bandwidth <- function(v, kernel = "qs", weights = NULL,
                                n = NULL) {
  ck <- .morie_check_kernel(kernel)
  q <- ck$const[1]; kq <- ck$const[2]; ik2 <- ck$const[3]
  t <- if (is.null(n)) nrow(v) else as.integer(n)
  al <- alpha_ar1(v, q = q, weights = weights)
  s <- (q * kq * kq * al$alpha * t / ik2) ^ (1 / (2 * q + 1))
  list(bandwidth = s, alpha = al$alpha, fits = al$fits)
}

#' Equation 2.3, kernel HAC on already-prewhitened vectors
#' @param v See Usage.
#' @param bandwidth See Usage.
#' @param kernel See Usage.
#' @param n_params See Usage.
#' @param n See Usage.
#' @export
kernel_hac <- function(v, bandwidth, kernel = "qs", n_params = 0L,
                       n = NULL) {
  ck <- .morie_check_kernel(kernel)
  kfun <- ck$fn; bounded <- as.logical(ck$const[4])
  rows <- as.matrix(v); storage.mode(rows) <- "double"
  m <- nrow(rows)
  if (m == 0L) stop("andmnh: no observations")
  p <- ncol(rows)
  t <- if (is.null(n)) m else as.integer(n)
  if (t <= n_params)
    stop("andmnh: T = ", t, " is not larger than the ", n_params,
         " estimated parameters")
  s <- as.numeric(bandwidth)
  if (s <= 0) stop("andmnh: bandwidth must be positive")
  jmax <- m - 1L
  if (bounded) jmax <- min(jmax, as.integer(floor(s)))
  out <- matrix(0, p, p)
  for (j in 0:jmax) {
    kj <- kfun(j / s)
    if (kj == 0) next
    gam <- matrix(0, p, p)
    for (tt in (j + 1L):m) {
      a <- rows[tt, ]; b <- rows[tt - j, ]
      for (i in seq_len(p)) if (a[i] != 0)
        for (k in seq_len(p)) gam[i, k] <- gam[i, k] + a[i] * b[k]
    }
    gam <- gam / t
    if (j == 0L) {
      out <- out + kj * gam
    } else {
      out <- out + kj * (gam + t(gam))
    }
  }
  dof <- t / (t - n_params)
  dof * out
}

#' The full VAR prewhitened kernel HAC estimator, eq. 2.4
#' @param e See Usage.
#' @param X See Usage.
#' @param prewhiten See Usage.
#' @param var_order See Usage.
#' @param kernel See Usage.
#' @param bandwidth See Usage.
#' @param weights See Usage.
#' @param n_params See Usage.
#' @param cap See Usage.
#' @param adjust See Usage.
#' @export
andrews_monahan_hac <- function(e, X = NULL, prewhiten = TRUE,
                                var_order = 1L, kernel = "qs",
                                bandwidth = NULL, weights = NULL,
                                n_params = NULL, cap = .EIGENVALUE_CAP,
                                adjust = TRUE) {
  if (!is.null(X)) {
    v <- moment_vectors(e, X)
    if (is.null(n_params)) n_params <- ncol(v)
  } else {
    v <- as.matrix(e); storage.mode(v) <- "double"
    if (is.null(n_params)) n_params <- 0L
  }
  n <- nrow(v)
  if (n == 0L) stop("andmnh: no observations")
  p <- ncol(v)
  order <- if (prewhiten) as.integer(var_order) else 0L
  pw <- prewhiten_var(v, order = order, cap = cap, adjust = adjust)
  if (is.null(bandwidth)) {
    ab <- automatic_bandwidth(pw$resid, kernel = kernel, weights = weights,
                              n = n)
    s <- ab$bandwidth; alpha <- ab$alpha; fits <- ab$fits; auto <- TRUE
  } else {
    s <- as.numeric(bandwidth); alpha <- NULL; fits <- NULL; auto <- FALSE
  }
  jstar <- kernel_hac(pw$resid, s, kernel = kernel, n_params = n_params, n = n)
  D <- as.matrix(pw$D)
  j <- D %*% jstar %*% t(D)
  list(J = j, J_star = jstar, D = D, A = pw$a_list, bandwidth = s,
       bandwidth_automatic = auto, alpha = alpha, ar1_fits = fits,
       kernel = kernel, var_order = order, n = n, p = p,
       n_params = as.integer(n_params), prewhitened = as.logical(order),
       method = "Andrews & Monahan (1992) VAR prewhitened kernel HAC, eq. 2.2-2.4, with the Andrews (1991) eq. 6.1 automatic bandwidth",
       note = sprintf("the VAR is a filter, not a model; its coefficients are capped through their SVD at %.2f so that I - sum(A_r) stays %.2f away from singular (footnote 4)", cap, 1 - cap))
}

#' Compact alias for andrews_monahan_hac
#' @export
andmnh <- andrews_monahan_hac

# house entry point: the package exports one morie_<module>
morie_andmnh <- andrews_monahan_hac
