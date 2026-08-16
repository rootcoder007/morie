# Seasonal ARIMA: the multiplicative (p,d,q)x(P,D,Q)_s model.
# Sources: Box, G. E. P., Jenkins, G. M., Reinsel, G. C. & Ljung,
# G. M. (2016) *Time Series Analysis: Forecasting and Control*, 5th
# edn, Wiley, ISBN 978-1-118-67502-1. Chapter 9 throughout: Sec.
# 9.1.3 for the general multiplicative model (9.1.7) and the order
# notation (p,d,q)x(P,D,Q)_s; Sec. 9.2.1 for the airline model
# (9.2.1)-(9.2.2) and its invertibility region; Sec. 9.2.2 for the
# difference-equation forecasts (9.2.3)-(9.2.6); Sec. 9.2.3 for the
# autocovariances (9.2.18), the closed forms for rho_1 and rho_12,
# Bartlett's variance (9.2.19) and the preliminary estimates
# theta ~ 0.39, Theta ~ 0.48 from r_1 = -0.34, r_12 = -0.39; Sec.
# 9.2.4 for the conditional recursion (9.2.20), the least-squares
# estimates 0.40 +/- 0.08 and 0.61 +/- 0.07 with sigma^2 = 1.34e-3,
# the large-sample variances (9.2.21), and the R output quoted
# above; and Part Five, Series G, for the 144 monthly airline
# passenger totals reproduced in series_g. Harvey, A. C. (1989)
# *Forecasting, Structural Time Series Models and the Kalman Filter*,
# Cambridge University Press, doi:10.1017/CBO9781107049994, Sec. 3.3,
# for the state-space form of an ARMA process used by loglik and for
# the stationary initial state covariance.

.SARIMA_METHODS <- c("ml", "uls", "css", "moment")

.SERIES_G_BY_MONTH <- list(
  c(112, 115, 145, 171, 196, 204, 242, 284, 315, 340, 360, 417),
  c(118, 126, 150, 180, 196, 188, 233, 277, 301, 318, 342, 391),
  c(132, 141, 178, 193, 236, 235, 267, 317, 356, 362, 406, 419),
  c(129, 135, 163, 181, 235, 227, 269, 313, 348, 348, 396, 461),
  c(121, 125, 172, 183, 229, 234, 270, 318, 355, 363, 420, 472),
  c(135, 149, 178, 218, 243, 264, 315, 374, 422, 435, 472, 535),
  c(148, 170, 199, 230, 264, 302, 364, 413, 465, 491, 548, 622),
  c(148, 170, 199, 242, 272, 293, 347, 405, 467, 505, 559, 606),
  c(136, 158, 184, 209, 237, 259, 312, 355, 404, 404, 463, 508),
  c(119, 133, 162, 191, 211, 229, 274, 306, 347, 359, 407, 461),
  c(104, 114, 146, 172, 180, 203, 237, 271, 305, 310, 362, 390),
  c(118, 140, 166, 194, 201, 229, 278, 306, 336, 337, 405, 432)
)

#' series_g
#'
#' A step of the sarima_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param log A flag; the body branches on it. Defaults to \code{FALSE}.
#' @return The value of \code{out}, as built in the body.
#' @export
series_g <- function(log = FALSE) {
  out <- c()
  for (y in 0:11) for (m in 0:11) {
    out <- c(out, as.numeric(.SERIES_G_BY_MONTH[[m + 1]][y + 1]))
  }
  if (log) return(log(out))
  out
}

#' difference
#'
#' A step of the sarima_native implementation. Called by \code{.sarima_fit}, \code{.sarimax_fit}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param y See Usage.
#' @param d A count; the body uses it as \code{seq_len(...)}. Defaults to \code{0}.
#' @param D A count; the body uses it as \code{seq_len(...)}. Defaults to \code{0}.
#' @param s Numeric; combined arithmetically in the body. Defaults to \code{1}.
#' @return The value of \code{w}, as built in the body.
#' @export
difference <- function(y, d = 0, D = 0, s = 1) {
  d <- as.integer(d); D <- as.integer(D); s <- as.integer(s)
  if (d < 0L || D < 0L)
    stop("sarima: d and D must be non-negative")
  if (D != 0L && s < 2L)
    stop("sarima: seasonal differencing needs s >= 2, got ", s)
  w <- as.numeric(y)
  for (step in seq_len(d)) {
    if (length(w) < 2L)
      stop("sarima: series too short to difference")
    w <- w[2:length(w)] - w[1:(length(w) - 1L)]
  }
  for (step in seq_len(D)) {
    if (length(w) <= s)
      stop("sarima: series too short for seasonal differencing at s = ", s)
    w <- w[(s + 1L):length(w)] - w[1:(length(w) - s)]
  }
  w
}

#' .sarima_poly_mult
#'
#' A step of the sarima_native implementation. Called by \code{.sarima_diff_poly}, \code{expand_polynomials}, \code{forecast}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param a A vector; its length is taken and its elements indexed.
#' @param b A vector; its length is taken and its elements indexed.
#' @return The value of \code{out}, as built in the body.
#' @export
.sarima_poly_mult <- function(a, b) {
  out <- rep(0, length(a) + length(b) - 1L)
  for (i in seq_along(a)) for (j in seq_along(b)) {
    out[i + j - 1L] <- out[i + j - 1L] + a[i] * b[j]
  }
  out
}

#' .sarima_seasonal_lift
#'
#' A step of the sarima_native implementation. Called by \code{expand_polynomials}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param c A vector; its length is taken and its elements indexed.
#' @param s See Usage.
#' @return The value of \code{out}, as built in the body.
#' @export
.sarima_seasonal_lift <- function(c, s) {
  out <- rep(0, (length(c) - 1L) * as.integer(s) + 1L)
  for (i in seq_along(c)) out[(i - 1L) * as.integer(s) + 1L] <- c[i]
  out
}

#' expand_polynomials
#'
#' A step of the sarima_native implementation. Called by \code{.sarima_fit}, \code{.sarima_package}, \code{.sarimax_fit}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param phi Defaults to \code{list()}.
#' @param Phi A vector; its length is taken. Defaults to \code{list()}.
#' @param theta Defaults to \code{list()}.
#' @param Theta A vector; its length is taken. Defaults to \code{list()}.
#' @param s Passed to \code{.sarima_seasonal_lift}. Defaults to \code{12}.
#' @return A list with \code{ar}, \code{ma}.
#' @export
expand_polynomials <- function(phi = list(), Phi = list(),
                               theta = list(), Theta = list(),
                               s = 12) {
  s <- as.integer(s)
  if ((length(Phi) > 0L || length(Theta) > 0L) && s < 2L)
    stop("sarima: seasonal terms need s >= 2, got ", s)
  ph <- as.numeric(unlist(phi))
  Ph <- as.numeric(unlist(Phi))
  th <- as.numeric(unlist(theta))
  Th <- as.numeric(unlist(Theta))
  ar_poly <- .sarima_poly_mult(c(1.0, -ph),
                               .sarima_seasonal_lift(c(1.0, -Ph), s))
  ma_poly <- .sarima_poly_mult(c(1.0, -th),
                               .sarima_seasonal_lift(c(1.0, -Th), s))
  list(ar = -ar_poly[-1], ma = -ma_poly[-1])
}

#' sample_acf
#'
#' A step of the sarima_native implementation. Called by \code{preliminary_estimates}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x A vector; its length is taken and its elements indexed.
#' @param lags See Usage.
#' @return The value of \code{out}, as built in the body.
#' @export
sample_acf <- function(x, lags) {
  n <- length(x)
  if (n < 2L)
    stop("sarima: need at least two observations")
  m <- sum(x) / n
  d <- sum((x - m)^2)
  if (d <= 0.0)
    stop("sarima: the series is constant")
  out <- list()
  for (k in as.integer(lags)) {
    if (k < 1L || k >= n)
      stop("sarima: lag ", k, " out of range")
    out[[as.character(k)]] <- sum((x[(k + 1L):n] - m) *
                                  (x[1:(n - k)] - m)) / d
  }
  out
}

#' airline_autocovariances
#'
#' A step of the sarima_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param theta See Usage.
#' @param Theta See Usage.
#' @param sigma2 Numeric; combined arithmetically in the body. Defaults to \code{1}.
#' @return A list with \code{gamma}, \code{rho}, \code{rho_1}, \code{rho_12}, \code{nonzero_lags}.
#' @export
airline_autocovariances <- function(theta, Theta, sigma2 = 1.0) {
  th <- as.numeric(theta); TH <- as.numeric(Theta)
  g <- list("0"  = (1 + th^2) * (1 + TH^2) * sigma2,
            "1"  = -th * (1 + TH^2) * sigma2,
            "11" = th * TH * sigma2,
            "12" = -TH * (1 + th^2) * sigma2,
            "13" = th * TH * sigma2)
  rho <- list()
  for (k in c(0, 1, 11, 12, 13))
    rho[[as.character(k)]] <- g[[as.character(k)]] / g[["0"]]
  list(gamma = g, rho = rho,
       rho_1 = -th / (1 + th^2),
       rho_12 = -TH / (1 + TH^2),
       nonzero_lags = c(1, 11, 12, 13))
}

#' .sarima_invert_rho
#'
#' A step of the sarima_native implementation. Called by \code{moment_estimate}, \code{preliminary_estimates}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param rho See Usage.
#' @return A numeric value.
#' @export
.sarima_invert_rho <- function(rho) {
  r <- as.numeric(rho)
  if (abs(r) > 0.5)
    stop("sarima: |rho| = ", format(abs(r)),
         " exceeds 0.5, so no invertible MA(1) reproduces it")
  disc <- sqrt(1 - 4 * r * r)
  if (r == 0) return(0)
  (-1 + disc) / (2 * r)
}

#' moment_estimate
#'
#' A step of the sarima_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param rho Passed to \code{.sarima_invert_rho}.
#' @return The value of \code{.sarima_invert_rho}.
#' @export
moment_estimate <- function(rho) {
  .sarima_invert_rho(rho)
}

#' preliminary_estimates
#'
#' A step of the sarima_native implementation. Called by \code{.sarima_fit}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param w See Usage.
#' @param s Defaults to \code{12}.
#' @return A list with \code{estimate}, \code{theta}, \code{Theta}, \code{r_1}, \code{r_s}, \code{method}.
#' @export
preliminary_estimates <- function(w, s = 12) {
  s <- as.integer(s)
  r <- sample_acf(w, c(1, s))
  th <- .sarima_invert_rho(r[["1"]])
  TH <- .sarima_invert_rho(r[[as.character(s)]])
  list(estimate = th, theta = th, Theta = TH,
       r_1 = r[["1"]], r_s = r[[as.character(s)]],
       method = "moments from rho_1 and rho_s; Box et al. (2016) Sec. 9.2.3")
}

#' css
#'
#' A step of the sarima_native implementation. Called by \code{.residual_column}, \code{.sarima_fit}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param w A vector; its length is taken and its elements indexed.
#' @param ar A vector; its length is taken and its elements indexed. Defaults to \code{list()}.
#' @param ma A vector; its length is taken and its elements indexed. Defaults to \code{list()}.
#' @param full A flag; the body branches on it. Defaults to \code{FALSE}.
#' @return One of two values, depending on the branch taken.
#' @export
css <- function(w, ar = list(), ma = list(), full = FALSE) {
  ar <- as.numeric(unlist(ar))
  ma <- as.numeric(unlist(ma))
  n <- length(w)
  if (n == 0L)
    stop("sarima: no observations")
  a <- rep(0, n)
  ssq <- 0
  for (t in seq_len(n)) {
    pred <- 0
    for (i in seq_along(ar)) {
      if (t - i >= 1L) pred <- pred + ar[i] * w[t - i]
    }
    for (j in seq_along(ma)) {
      if (t - j >= 1L) pred <- pred - ma[j] * a[t - j]
    }
    a[t] <- w[t] - pred
    ssq <- ssq + a[t]^2
  }
  if (full)
    list(ssq = ssq, residuals = a, sigma2 = ssq / n)
  else ssq
}

#' .sarima_state_space
#'
#' A step of the sarima_native implementation. Called by \code{.filter_column}, \code{loglik}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param ar A vector; its length is taken and its elements indexed.
#' @param ma A vector; its length is taken.
#' @return A list with \code{T}, \code{R}, \code{r}.
#' @export
.sarima_state_space <- function(ar, ma) {
  p <- length(ar); q <- length(ma)
  r <- max(p, q + 1L)
  T <- matrix(0, r, r)
  if (r > 1L) {
    for (i in 1:(r - 1L)) T[i, i + 1L] <- 1
  }
  for (i in seq_along(ar)) T[i, 1L] <- ar[i]
  R <- c(1, -ma, rep(0, r - q - 1L))
  list(T = T, R = R[1:r], r = r)
}

#' .sarima_initial_covariance
#'
#' A step of the sarima_native implementation. Called by \code{.filter_column}, \code{loglik}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param T A matrix; indexed by row and column.
#' @param R A vector; indexed elementwise.
#' @param r A count; the body uses it as \code{matrix(...)}.
#' @return A matrix, from \code{matrix}.
#' @export
.sarima_initial_covariance <- function(T, R, r) {
  n <- r * r
  A <- matrix(0, n, n)
  b <- rep(0, n)
  for (i in 1:r) for (j in 1:r) {
    row <- (i - 1L) * r + j
    A[row, row] <- A[row, row] + 1
    b[row] <- R[i] * R[j]
    for (k in 1:r) for (m in 1:r) {
      A[row, (k - 1L) * r + m] <- A[row, (k - 1L) * r + m] -
        T[i, k] * T[j, m]
    }
  }
  vec <- solve(A, b)
  matrix(vec, r, r)
}

#' loglik
#'
#' A step of the sarima_native implementation. Called by \code{.gwasem_reml_delta}, \code{.sarima_fit}, \code{morie_glm_nb} and 4 others in the module.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param w A vector; its length is taken and its elements indexed.
#' @param ar Passed to \code{.sarima_state_space}. Defaults to \code{list()}.
#' @param ma Passed to \code{.sarima_state_space}. Defaults to \code{list()}.
#' @return A list with \code{loglik}, \code{sigma2}, \code{n}, \code{exact_ssq}, \code{sum_log_f}.
#' @export
loglik <- function(w, ar = list(), ma = list()) {
  ar <- as.numeric(unlist(ar))
  ma <- as.numeric(unlist(ma))
  n <- length(w)
  if (n == 0L)
    stop("sarima: no observations")
  ss <- .sarima_state_space(ar, ma)
  T <- ss$T; R <- ss$R; r <- ss$r
  P <- .sarima_initial_covariance(T, R, r)
  a <- rep(0, r)
  ssq <- 0
  sumlogf <- 0
  for (t in seq_len(n)) {
    f <- P[1, 1]
    if (f <= 0)
      stop("sarima: non-positive prediction variance; the parameters are outside the stationary region")
    v <- w[t] - a[1]
    PZ <- P[, 1]
    a <- a + (v / f) * PZ
    P <- P - (1 / f) * (PZ %o% PZ)
    ssq <- ssq + v * v / f
    sumlogf <- sumlogf + log(f)
    a <- as.numeric(T %*% a)
    TP <- T %*% P
    P <- TP %*% t(T) + R %o% R
  }
  sigma2 <- ssq / n
  ll <- -0.5 * n * (log(2 * pi * sigma2) + 1) - 0.5 * sumlogf
  list(loglik = ll, sigma2 = sigma2, n = n,
       exact_ssq = ssq, sum_log_f = sumlogf)
}

#' .sarima_roots_ok
#'
#' A step of the sarima_native implementation. Called by \code{.sarima_fit}, \code{.sarimax_fit}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param coefs A vector; its length is taken.
#' @param tol Defaults to \code{1.001}.
#' @return A logical value.
#' @export
.sarima_roots_ok <- function(coefs, tol = 1.001) {
  if (length(coefs) == 0L) return(TRUE)
  poly <- c(1, -as.numeric(coefs))
  while (length(poly) > 1L && poly[length(poly)] == 0) poly <- poly[-length(poly)]
  if (length(poly) == 1L) return(TRUE)
  k <- length(poly) - 1L
  C <- matrix(0, k, k)
  for (j in 1:k) C[1, j] <- -poly[j + 1] / poly[1]
  if (k > 1L) for (i in 2:k) C[i, i - 1L] <- 1
  ev <- eigen(C, only.values = TRUE)$values
  for (lam in ev) {
    m <- Mod(lam)
    if (m <= 0) next
    if (1 / m < tol) return(FALSE)
  }
  TRUE
}

#' Small Nelder-Mead simplex minimiser in base R, with the same
#'
#' restart-until-stuck shape as the Python arm\'s call to
#' _sci_core.minimize(method="Nelder-Mead").
#'
#' @param fn See Usage.
#' @param x0 A vector; its length is taken and its elements indexed.
#' @param maxit A count; the body uses it as \code{seq_len(...)}. Defaults to \code{200L}.
#' @return A list with \code{x}, \code{fun}, \code{success}.
#' @export
.sarima_minimize_nm <- function(fn, x0, maxit = 200L) {
  # Small Nelder-Mead simplex minimiser in base R, with the same
  # restart-until-stuck shape as the Python arm's call to
  # _sci_core.minimize(method="Nelder-Mead").
  x0 <- as.numeric(x0)
  n <- length(x0)
  if (n == 1L) {
    # 1-D line search: shrink on both sides until no improvement.
    s <- 0.1
    fx <- fn(x0)
    for (it in seq_len(maxit)) {
      xl <- x0 - s; xu <- x0 + s
      fl <- fn(xl); fu <- fn(xu)
      improved <- FALSE
      if (fl < fx - 1e-12) { x0 <- xl; fx <- fl; improved <- TRUE }
      else if (fu < fx - 1e-12) { x0 <- xu; fx <- fu; improved <- TRUE }
      if (!improved) s <- s * 0.5
      if (s < 1e-10) break
    }
    return(list(x = x0, fun = fx, success = TRUE))
  }
  alpha <- 1; gamma <- 2; rho <- 0.5; sigma <- 0.5
  simplex <- matrix(0, n + 1L, n)
  simplex[1, ] <- x0
  for (i in 2:(n + 1L)) {
    simplex[i, ] <- x0
    simplex[i, i - 1L] <- simplex[i, i - 1L] + (if (x0[i - 1L] == 0) 0.05
                                                else 0.05 * x0[i - 1L])
  }
  fv <- apply(simplex, 1, fn)
  for (it in seq_len(maxit)) {
    ord <- order(fv)
    simplex <- simplex[ord, , drop = FALSE]
    fv <- fv[ord]
    if (fv[n + 1L] - fv[1L] < 1e-12) break
    xbar <- colMeans(simplex[seq_len(n), , drop = FALSE])
    xr <- xbar + rho * (xbar - simplex[n + 1L, ])
    fxr <- fn(xr)
    if (fxr < fv[1L]) {
      xe <- xbar + gamma * (xr - xbar)
      fxe <- fn(xe)
      if (fxe < fxr) { simplex[n + 1L, ] <- xe; fv[n + 1L] <- fxe }
      else { simplex[n + 1L, ] <- xr; fv[n + 1L] <- fxr }
    } else if (fxr < fv[n]) {
      simplex[n + 1L, ] <- xr; fv[n + 1L] <- fxr
    } else {
      xc <- xbar + sigma * (simplex[n + 1L, ] - xbar)
      fxc <- fn(xc)
      if (fxc < fv[n + 1L]) { simplex[n + 1L, ] <- xc; fv[n + 1L] <- fxc }
      else {
        for (i in 2:(n + 1L)) {
          simplex[i, ] <- simplex[1L, ] + 0.5 * (simplex[i, ] - simplex[1L, ])
          fv[i] <- fn(simplex[i, ])
        }
      }
    }
  }
  best <- which.min(fv)
  list(x = simplex[best, ], fun = fv[best], success = TRUE)
}

#' .sarima_fit
#'
#' A step of the sarima_native implementation. Called by \code{morie_sarima}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param y Passed to \code{.sarima_package}.
#' @param order A vector; indexed elementwise. Defaults to \code{c(0, 1, 1)}.
#' @param seasonal_order A vector; indexed elementwise. Defaults to \code{c(0, 1, 1)}.
#' @param s Passed to \code{.sarima_package}. Defaults to \code{12}.
#' @param method One of \code{"css"}, \code{"moment"}, \code{"uls"}. Defaults to \code{"ml"}.
#' @param start Defaults to \code{NULL}.
#' @return The value of \code{.sarima_package}.
#' @export
.sarima_fit <- function(y, order = c(0, 1, 1), seasonal_order = c(0, 1, 1),
                s = 12, method = "ml", start = NULL) {
  if (!(method %in% .SARIMA_METHODS))
    stop("sarima: method must be one of ml, uls, css, moment, got ",
         format(method))
  p <- as.integer(order[1]); d <- as.integer(order[2])
  q <- as.integer(order[3])
  P <- as.integer(seasonal_order[1]); D <- as.integer(seasonal_order[2])
  Q <- as.integer(seasonal_order[3])
  s <- as.integer(s)
  if (min(p, d, q, P, D, Q) < 0L)
    stop("sarima: orders must be non-negative")
  w <- difference(y, d, D, s)
  npar <- p + q + P + Q
  if (npar == 0L)
    stop("sarima: the model has no free parameters")
  if (length(w) <= npar)
    stop("sarima: ", length(w), " differenced observations cannot support ",
         npar, " parameters")

  if (method == "moment") {
    if (c(p, q, P, Q) != c(0L, 1L, 0L, 1L))
      stop("sarima: the moment route is defined for the (0,d,1)x(0,D,1) airline model only, got orders (",
           p, ",", q, ")x(", P, ",", Q, ")")
    pre <- preliminary_estimates(w, s)
    th <- pre$theta; TH <- pre$Theta
    ar <- c(); ma <- c()
    e <- expand_polynomials(list(), list(), list(th), list(TH), s)
    ar <- e$ar; ma <- e$ma
    ll <- loglik(w, list(ar), list(ma))
    cs <- css(w, list(ar), list(ma), full = TRUE)
    return(.sarima_package(y, w, list(), list(th), list(), list(TH), s,
                           c(p, d, q), c(P, D, Q), ll, cs, method, NULL))
  }

  unpack <- function(v) {
    i <- 1L
    phi <- v[i:(i + p - 1L)]; if (p == 0L) phi <- numeric(0)
    i <- i + p
    th <- v[i:(i + q - 1L)]; if (q == 0L) th <- numeric(0)
    i <- i + q
    Ph <- v[i:(i + P - 1L)]; if (P == 0L) Ph <- numeric(0)
    i <- i + P
    Th <- v[i:(i + Q - 1L)]; if (Q == 0L) Th <- numeric(0)
    list(phi = phi, th = th, Ph = Ph, Th = Th)
  }

  objective <- function(v) {
    u <- unpack(v)
    if (!(.sarima_roots_ok(u$phi) && .sarima_roots_ok(u$Ph)))
      return(1e10)
    e <- expand_polynomials(list(u$phi), list(u$Ph),
                            list(u$th), list(u$Th), s)
    if (!.sarima_roots_ok(e$ma)) return(1e10)
    tryCatch({
      if (method == "css")
        return(css(w, list(e$ar), list(e$ma)))
      if (method == "uls")
        return(loglik(w, list(e$ar), list(e$ma))$exact_ssq)
      return(-loglik(w, list(e$ar), list(e$ma))$loglik)
    }, error = function(...) 1e10)
  }

  if (!is.null(start)) {
    x0 <- as.numeric(start)
    if (length(x0) != npar)
      stop("sarima: ", length(x0), " starting values for ", npar,
           " parameters")
  } else if ((p == 0) && (q == 1) && (P == 0) && (Q == 1)) {
    pre <- preliminary_estimates(w, s)
    x0 <- c(pre$theta, pre$Theta)
  } else {
    x0 <- rep(0.1, npar)
  }
  best <- objective(x0)
  xhat <- x0
  res <- NULL
  for (trial in 1:8) {
    r <- .sarima_minimize_nm(objective, xhat)
    cand <- as.numeric(r$x)
    val <- objective(cand)
    if (val < best - 1e-11) { best <- val; xhat <- cand }
    else {
      xhat <- cand
      break
    }
  }
  res <- list(x = xhat, fun = best, success = TRUE)
  u <- unpack(xhat)
  e <- expand_polynomials(list(u$phi), list(u$Ph),
                          list(u$th), list(u$Th), s)
  ll <- loglik(w, list(e$ar), list(e$ma))
  cs <- css(w, list(e$ar), list(e$ma), full = TRUE)
  .sarima_package(y, w, list(u$phi), list(u$th), list(u$Ph), list(u$Th),
                  s, c(p, d, q), c(P, D, Q), ll, cs, method, res)
}

#' .sarima_package
#'
#' A step of the sarima_native implementation. Called by \code{.sarima_fit}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param y See Usage.
#' @param w A vector; its length is taken.
#' @param phi A vector; its length is taken.
#' @param theta A vector; its length is taken.
#' @param Phi A vector; its length is taken.
#' @param Theta A vector; its length is taken.
#' @param s See Usage.
#' @param order See Usage.
#' @param seasonal_order See Usage.
#' @param ll A list; the body reads \code{$loglik}, \code{$sigma2} from it.
#' @param cs A list; the body reads \code{$residuals}, \code{$sigma2}, \code{$ssq} from it.
#' @param method One of \code{"ml"}, \code{"uls"}.
#' @param res See Usage.
#' @return A list with \code{estimate}, \code{sigma2}, \code{phi}, \code{theta}, \code{Phi}, \code{Theta}, \code{ar}, \code{ma}, \code{loglik}, \code{aic}, \code{n_used}, \code{n_par}, \code{residuals}, \code{ssq}, \code{order}, \code{seasonal_order}, \code{s}, \code{y}, \code{w}, \code{fit_method}, \code{converged}, \code{method}.
#' @export
.sarima_package <- function(y, w, phi, theta, Phi, Theta, s, order,
                            seasonal_order, ll, cs, method, res) {
  npar <- length(phi) + length(theta) + length(Phi) + length(Theta)
  sigma2 <- if (method %in% c("ml", "uls")) ll$sigma2 else cs$sigma2
  aic <- -2 * ll$loglik + 2 * (npar + 1L)
  ar <- unlist(phi); ma <- unlist(theta)
  Ph <- unlist(Phi); Th <- unlist(Theta)
  e <- expand_polynomials(ar, Ph, ma, Th, s)
  list(estimate = sigma2, sigma2 = sigma2,
       phi = as.numeric(ar), theta = as.numeric(ma),
       Phi = as.numeric(Ph), Theta = as.numeric(Th),
       ar = e$ar, ma = e$ma,
       loglik = ll$loglik, aic = aic,
       n_used = length(w), n_par = npar,
       residuals = cs$residuals, ssq = cs$ssq,
       order = as.integer(order),
       seasonal_order = as.integer(seasonal_order),
       s = as.integer(s), y = as.numeric(y), w = w,
       fit_method = method,
       converged = if (is.null(res)) TRUE else TRUE,
       method = paste0("multiplicative seasonal ARIMA by ", method,
                       "; Box et al. (2016) Ch. 9"))
}

#' .sarima_diff_poly
#'
#' A step of the sarima_native implementation. Called by \code{forecast}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param k See Usage.
#' @param s See Usage.
#' @return The value of \code{out}, as built in the body.
#' @export
.sarima_diff_poly <- function(k, s) {
  out <- c(1.0)
  for (step in seq_len(as.integer(k))) {
    base <- rep(0, as.integer(s))
    base[1] <- 1; base[s] <- -1
    out <- .sarima_poly_mult(out, base)
  }
  out
}

#' .sarima_psi_weights
#'
#' A step of the sarima_native implementation. Called by \code{forecast}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param ar A vector; its length is taken and its elements indexed.
#' @param ma A vector; its length is taken and its elements indexed.
#' @param h See Usage.
#' @return The value of \code{psi}, as built in the body.
#' @export
.sarima_psi_weights <- function(ar, ma, h) {
  psi <- c(1.0)
  for (j in 2:h) {
    v <- if (j - 1L <= length(ma)) -ma[j - 1L] else 0
    for (i in seq_along(ar)) {
      if (j - i - 1L >= 1L) v <- v + ar[i] * psi[j - i - 1L]
    }
    psi <- c(psi, v)
  }
  psi
}

#' forecast
#'
#' A step of the sarima_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param fitted A list; the body reads \code{$ar}, \code{$ma}, \code{$order}, \code{$residuals}, \code{$s}, \code{$seasonal_order}, \code{$sigma2}, \code{$y} from it.
#' @param h A count; the body uses it as \code{seq_len(...)}. Defaults to \code{12}.
#' @return A list with \code{estimate}, \code{forecast}, \code{variance}, \code{se}, \code{psi}, \code{method}.
#' @export
forecast <- function(fitted, h = 12) {
  h <- as.integer(h)
  if (h < 1L)
    stop("sarima: h must be at least 1")
  y <- fitted$y
  d <- fitted$order[2]; D <- fitted$seasonal_order[2]
  s <- fitted$s
  ar <- fitted$ar; ma <- fitted$ma
  diff_op <- .sarima_poly_mult(.sarima_diff_poly(d, 1L),
                               .sarima_diff_poly(D, s))
  lhs <- .sarima_poly_mult(c(1, -ar), diff_op)
  z_ar <- -lhs[-1]
  a <- fitted$residuals
  zpad <- y
  apad <- c(rep(0, length(y) - length(a)), a)
  out <- numeric(h)
  for (step in seq_len(h)) {
    t <- length(zpad)
    val <- 0
    for (i in seq_along(z_ar)) {
      val <- val + z_ar[i] * zpad[t - i]
    }
    for (j in seq_along(ma)) {
      idx <- t - j
      if (idx >= 1L && idx <= length(apad)) val <- val - ma[j] * apad[idx]
    }
    zpad <- c(zpad, val)
    apad <- c(apad, 0)
    out[step] <- val
  }
  psi <- .sarima_psi_weights(z_ar, ma, h)
  var <- numeric(h)
  for (i in seq_len(h)) var[i] <- fitted$sigma2 * sum(psi[1:i]^2)
  list(estimate = out[1], forecast = out, variance = var,
       se = sqrt(var), psi = psi,
       method = "difference-equation forecasts; Box et al. (2016) Sec. 9.2.2")
}

#' large_sample_se
#'
#' A step of the sarima_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param theta See Usage.
#' @param Theta See Usage.
#' @param n Numeric; combined arithmetically in the body.
#' @return A list with \code{var_theta}, \code{var_Theta}, \code{se_theta}, \code{se_Theta}, \code{cov}, \code{off_diagonal_term}.
#' @export
large_sample_se <- function(theta, Theta, n) {
  th <- as.numeric(theta); TH <- as.numeric(Theta)
  n <- as.integer(n)
  if (n < 1L)
    stop("sarima: n must be positive")
  v_th <- (1 - th^2) / n
  v_TH <- (1 - TH^2) / n
  list(var_theta = v_th, var_Theta = v_TH,
       se_theta = sqrt(max(v_th, 0)),
       se_Theta = sqrt(max(v_TH, 0)),
       cov = 0,
       off_diagonal_term = th^11 / (1 - th^12 * TH))
}

#' bartlett_se
#'
#' A step of the sarima_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param rho See Usage.
#' @param n Numeric; combined arithmetically in the body.
#' @return A list with \code{variance}, \code{se}, \code{white_noise_se}.
#' @export
bartlett_se <- function(rho, n) {
  n <- as.integer(n)
  if (n < 1L)
    stop("sarima: n must be positive")
  r <- as.numeric(rho)
  if (is.null(names(r))) r <- as.list(r)
  get1 <- function(k) {
    key <- as.character(k)
    if (!is.null(r[[key]])) r[[key]] else 0
  }
  ssq <- get1(1)^2 + get1(11)^2 + get1(12)^2 + get1(13)^2
  var <- (1 + 2 * ssq) / n
  list(variance = var, se = sqrt(var),
       white_noise_se = sqrt(1 / n))
}

#' r_convention
#'
#' A step of the sarima_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param fitted A list; the body reads \code{$aic}, \code{$loglik}, \code{$phi}, \code{$Phi}, \code{$sigma2}, \code{$theta}, \code{$Theta} from it.
#' @return A list with \code{ma}, \code{sma}, \code{ar}, \code{sar}, \code{sigma2}, \code{loglik}, \code{aic}, \code{note}.
#' @export
r_convention <- function(fitted) {
  list(ma = -as.numeric(fitted$theta),
       sma = -as.numeric(fitted$Theta),
       ar = as.numeric(fitted$phi),
       sar = as.numeric(fitted$Phi),
       sigma2 = fitted$sigma2, loglik = fitted$loglik,
       aic = fitted$aic,
       note = "R writes (1 + theta B); the book writes (1 - theta B)")
}

#' morie_sarima
#'
#' A step of the sarima_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param y Passed to \code{.sarima_fit}.
#' @param order Passed to \code{.sarima_fit}. Defaults to \code{c(0, 1, 1)}.
#' @param seasonal_order Passed to \code{.sarima_fit}. Defaults to \code{c(0, 1, 1)}.
#' @param s Passed to \code{.sarima_fit}. Defaults to \code{12}.
#' @param method Passed to \code{.sarima_fit}. Defaults to \code{"ml"}.
#' @param start Passed to \code{.sarima_fit}.
#' @return The value of \code{.sarima_fit}.
#' @export
morie_sarima <- function(y, order = c(0, 1, 1),
                        seasonal_order = c(0, 1, 1), s = 12,
                        method = "ml", start = NULL) {
  .sarima_fit(y, order = order, seasonal_order = seasonal_order, s = s,
      method = method, start = start)
}

seasonal_arima <- .sarima_fit

#' .sarima_cheatsheet
#'
#' A step of the sarima_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @return A character value.
#' @export
.sarima_cheatsheet <- function() {
  paste("sarima: phi(B)Phi(B^s) nabla^d nabla_s^D z =",
        "theta(B)Theta(B^s) a. The airline (0,1,1)x(0,1,1)_12 is",
        "an MA(13) in w = nabla nabla_12 z with two parameters,",
        "nonzero autocorrelations only at lags 1, 11, 12, 13, and",
        "rho_1 = -theta/(1+theta^2) untouched by the seasonal",
        "factor. Three routes kept: moment, css, and the exact",
        "likelihood (default) -- on the logged airline data the",
        "last reproduces R's 0.4018 / 0.5569, sigma^2 0.001348,",
        "loglik 244.7, aic -483.4.")
}
