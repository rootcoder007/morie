# ABC with a Gaussian-process surrogate.
# Sources: Wilkinson, R. D. (2014) "Accelerating ABC methods using
# Gaussian processes", AISTATS PMLR 33, 1015-1023 -- the GP-on-log-
# likelihood emulator and the sequential history matching of Sec. 3,
# with the implausibility rule m + 3 sigma < max_E l_hat - T at T = 10.
# Meeds, E. & Welling, M. (2014) "GPS-ABC: Gaussian process surrogate
# approximate Bayesian computation", UAI 2014, 593-602 -- the adaptive
# synthetic-likelihood sampler (eqs. 9-16, Algorithms 1-2), the
# synthetic likelihood also appears in Wilkinson eq. (2) and in
# Wood, S. N. (2010), Nature 466, 1102-1104.
# Sobol, I. M. (1967) USSR Comp. Math. Math. Phys. 7(4), 86-112 and
# Bratley, P. & Fox, B. L. (1988) ACM TOMS 14(1), 88-100 -- the
# direction numbers and Gray-code recurrence used in design_from_prior.
# Craig, P. S., Goldstein, M., Seheult, A. H. & Smith, J. A. (1997),
# Case Studies in Bayesian Statistics III, Springer LNS 121, 37-93 --
# the history-matching waves.
#
# Native implementation mirroring morie.fn.abcgp exactly: the same
# Sobol points, the same GP posterior with the empirical Bayes
# (beta, tau^2) integrated out under the improper 1/tau^2 prior, the
# same GABC Monte-Carlo likelihood and bootstrap nugget, the same
# history matching, the same Meeds-Welling synthetic likelihood and
# adaptive threshold. All random numbers are pulled from the shared
# .ghc_rng so the two arms produce the same stream.

.SOBOL_POLY <- list(
  NULL,                       # dimension 1: no polynomial, m_k = 1
  list(degree = 1L, coeff = 0L, m_init = c(1)),
  list(degree = 2L, coeff = 1L, m_init = c(1, 3)),
  list(degree = 3L, coeff = 1L, m_init = c(1, 3, 1)),
  list(degree = 3L, coeff = 2L, m_init = c(1, 1, 1)),
  list(degree = 4L, coeff = 1L, m_init = c(1, 1, 3, 3)),
  list(degree = 4L, coeff = 4L, m_init = c(1, 3, 5, 13)),
  list(degree = 5L, coeff = 2L, m_init = c(1, 1, 5, 5, 17))
)

.MORIE_GP_KERNELS <- c("sqexp", "matern32", "matern52")
.MORIE_GP_METHODS <- c("wilkinson", "gps", "adaptive", "synthetic")

# log(sum(exp(v))) with Wilkinson's footnote-1 shift. Drops non-finite
# entries: any single -inf would otherwise dominate the sum.
#' Log(sum(exp(v))) with Wilkinson\'s footnote-1 shift. Drops non-finite
#'
#' entries: any single -inf would otherwise dominate the sum.
#'
#' @param values A vector; indexed elementwise.
#' @return A numeric value.
#' @export
#' @examples
#' x <- c(1.2, 2.4, 3.1, 4.8, 5.3, 6.7, 7.1, 8.9)
#' res <- .gp_lse(values = x)
#' res
.gp_lse <- function(values) {
  vals <- values[is.finite(values)]
  if (length(vals) == 0L) return(-Inf)
  a <- max(vals)
  if (a == -Inf) return(-Inf)
  a + log(sum(exp(vals - a)))
}

#' Sobol low-discrepancy sequence in [0, 1)^dim
#'
#' The first \code{n} points of the Sobol sequence in the requested
#' dimension. The design is shared with the Python arm down to the
#' last bit and is the design Wilkinson (2014) Sec. 2.2 uses.
#'
#' @param n Number of points, integer >= 1.
#' @param dim Dimension; must be in 1..8 (the tabulated direction
#'   numbers).
#' @param skip Drop that many leading points.
#' @return Numeric matrix with \code{n} rows and \code{dim} columns.
#' @references Sobol, I. M. (1967). Bratley, P. & Fox, B. L. (1988).
#' @export
sobol_sequence <- function(n, dim, skip = 0L) {
  n <- as.integer(n)
  dim <- as.integer(dim)
  if (n < 1L) stop("sobol_sequence: n must be at least 1")
  if (dim < 1L || dim > length(.SOBOL_POLY))
    stop("sobol_sequence: dim must be between 1 and ",
         length(.SOBOL_POLY))
  total <- n + as.integer(skip)
  bits <- max(1L, as.integer(ceiling(log(total + 1, base = 2))) + 1L)
  v <- vector("list", dim)
  for (d in seq_len(dim)) {
    entry <- .SOBOL_POLY[[d]]
    if (is.null(entry)) {
      m <- rep(1L, bits)
    } else {
      degree <- entry$degree
      coeff <- entry$coeff
      m <- as.integer(entry$m_init)
      for (k in (degree + 1L):bits) {
        val <- bitwXor(m[k - degree], bitwShiftL(m[k - degree], degree))
        for (j in seq_len(degree - 1L)) {
          if (bitwAnd(bitwShiftR(coeff, degree - 1L - j), 1L) == 1L)
            val <- bitwXor(val, bitwShiftL(m[k - j], j))
        }
        m <- c(m, val)
      }
    }
    v[[d]] <- as.numeric(m) * (2 ^ (seq.int(bits, 1L, by = -1L) - 1L))
  }
  out <- matrix(0, nrow = n, ncol = dim)
  x <- integer(dim)
  denom <- 2 ^ bits
  for (i in 0L:(total - 1L)) {
    if (i >= as.integer(skip)) out[i + 1L - as.integer(skip), ] <- x / denom
    c <- 0L
    value <- i
    while (bitwAnd(value, 1L) == 1L) { value <- bitwShiftR(value, 1L)
    c <- c + 1L }
    for (d in seq_len(dim)) x[d] <- bitwXor(x[d], v[[d]][c + 1L])
  }
  out
}

#' Sobol design pushed through the prior's inverse CDF
#'
#' @param n Number of points.
#' @param prior_ppf Either a list of per-parameter quantile functions
#'   or a list \code{list(lo, hi)} for independent uniform priors.
#' @param dim Optional dimension check.
#' @param skip Drop that many leading Sobol points (default 1, so the
#'   origin is not sent to the prior corner).
#' @return Numeric matrix with \code{n} rows.
#' @export
design_from_prior <- function(n, prior_ppf, dim = NULL, skip = 1L) {
  if (is.list(prior_ppf) && length(prior_ppf) == 2L &&
        !is.function(prior_ppf[[1]])) {
    lo <- as.numeric(prior_ppf[[1]])
    hi <- as.numeric(prior_ppf[[2]])
    if (length(lo) != length(hi))
      stop("design_from_prior: lo and hi differ in length")
    u <- sobol_sequence(n, length(lo), skip = skip)
    out <- matrix(0, nrow = n, ncol = length(lo))
    for (j in seq_along(lo))
      out[, j] <- lo[j] + (hi[j] - lo[j]) * u[, j]
    return(out)
  }
  fns <- prior_ppf
  if (!is.null(dim) && as.integer(dim) != length(fns))
    stop("design_from_prior: dim does not match number of quantile functions")
  u <- sobol_sequence(n, length(fns), skip = skip)
  out <- matrix(0, nrow = n, ncol = length(fns))
  for (j in seq_along(fns)) out[, j] <- as.numeric(fns[[j]](u[, j]))
  out
}

#' .gp_summarise
#'
#' A step of the abcgp_native implementation. Called by \code{.gp_mw_sampler},
#' \code{gabc_log_likelihood}, \code{synthetic_log_likelihood}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x Passed to \code{summary}.
#' @param summary Optional; may be \code{NULL}. Passed to \code{is.null}.
#' @return A vector, from \code{as.numeric}.
#' @export
.gp_summarise <- function(x, summary) {
  v <- if (is.null(summary)) x else summary(x)
  as.numeric(v)
}

#' Wilkinson's GABC log-likelihood
#'
#' Monte Carlo estimate of the GABC likelihood of Wilkinson (2014)
#' eq. (1) and the bootstrap variance that is used as the GP's nugget.
#'
#' @param sim Simulator, signature \code{(theta, rng)}.
#' @param obs Observed data.
#' @param theta Parameter vector.
#' @param n_sim Number of Monte Carlo samples.
#' @param epsilon Acceptance tolerance.
#' @param summary Optional summary function.
#' @param kernel Acceptance kernel, "gaussian" or "uniform".
#' @param seed Random seed.
#' @param bootstrap Number of bootstrap replications for the nugget.
#' @return List \code{log_lik, nugget_variance}.
#' @export
gabc_log_likelihood <- function(sim, obs, theta, n_sim = 50L, epsilon = 1.0,
                                summary = NULL, kernel = "gaussian",
                                seed = 0L, bootstrap = 25L) {
  if (!(kernel %in% c("gaussian", "uniform")))
    stop("gabc_log_likelihood: kernel must be 'gaussian' or 'uniform'")
  eps <- as.numeric(epsilon)
  if (eps <= 0) stop("gabc_log_likelihood: epsilon must be positive")
  d_obs <- .gp_summarise(obs, summary)
  e <- .ghc_rng(seed)
  terms <- numeric(0)
  m <- as.integer(n_sim)
  for (k in seq_len(m)) {
    x <- sim(theta, e)
    s <- .gp_summarise(x, summary)
    if (length(s) != length(d_obs))
      stop("gabc_log_likelihood: simulator summary length mismatch")
    rho <- sqrt(sum((s - d_obs) ^ 2))
    if (kernel == "uniform") {
      terms <- c(terms, if (rho <= eps) 0 else -Inf)
    } else {
      terms <- c(terms, -0.5 * (rho / eps) ^ 2)
    }
  }
  log_lik <- .gp_lse(terms) - log(m)
  reps <- as.integer(bootstrap)
  if (reps < 2L || m < 2L) return(list(log_lik = log_lik, nugget_variance = 0))
  boot <- numeric(reps)
  for (b in seq_len(reps)) {
    idx <- pmin(as.integer(.ghc_unif(e, m) * m), m - 1L)
    boot[b] <- .gp_lse(terms[idx + 1L]) - log(m)
  }
  finite <- boot[is.finite(boot)]
  if (length(finite) < 2L) return(list(log_lik = log_lik, nugget_variance = 0))
  mu <- mean(finite)
  list(log_lik = log_lik,
       nugget_variance = sum((finite - mu) ^ 2) / (length(finite) - 1L))
}

#' .gp_chol
#'
#' A step of the abcgp_native implementation. Called by \code{.gp_draw_mean},
#' \code{.gp_mvn_logpdf}, \code{.gp_profile_nll} and 1 others in the module.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param a A matrix; indexed by row and column.
#' @param jitter Defaults to \code{1e-12}.
#' @return The value of \code{L}, as built in the body.
#' @export
#' @examples
#' A <- matrix(c(4, 1, 0.5, 1, 3, 0.8, 0.5, 0.8, 2), nrow = 3)
#' res <- .gp_chol(a = A)
#' res
.gp_chol <- function(a, jitter = 1e-12) {
  n <- nrow(a)
  L <- matrix(0, n, n)
  for (i in seq_len(n)) {
    for (j in seq_len(i)) {
      s <- a[i, j] - if (j > 1L) sum(L[i, 1:(j - 1L)] * L[j, 1:(j - 1L)]) else 0
      if (i == j) {
        if (s <= 0) s <- jitter
        L[i, j] <- sqrt(s)
      } else {
        L[i, j] <- s / L[j, j]
      }
    }
  }
  L
}

#' .gp_chol_solve
#'
#' A step of the abcgp_native implementation. Called by \code{.gp_mvn_logpdf},
#' \code{.gp_profile_nll}, \code{gp_fit} and 1 others in the module.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param L A matrix; indexed by row and column.
#' @param b A vector; indexed elementwise.
#' @return The value of \code{x}, as built in the body.
#' @export
.gp_chol_solve <- function(L, b) {
  n <- nrow(L)
  y <- numeric(n)
  for (i in seq_len(n)) {
    s <- if (i > 1L) sum(L[i, 1:(i - 1L)] * y[1:(i - 1L)]) else 0
    y[i] <- (b[i] - s) / L[i, i]
  }
  x <- numeric(n)
  for (i in n:1L) {
    s <- if (i < n) sum(L[(i + 1L):n, i] * x[(i + 1L):n]) else 0
    x[i] <- (y[i] - s) / L[i, i]
  }
  x
}

#' .gp_mvn_logpdf
#'
#' A step of the abcgp_native implementation. Called by \code{.gp_mw_sampler},
#' \code{synthetic_log_likelihood}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param y A vector; its length is taken.
#' @param mu Numeric; combined arithmetically in the body.
#' @param cov Passed to \code{.gp_chol}.
#' @return A numeric value.
#' @export
.gp_mvn_logpdf <- function(y, mu, cov) {
  n <- length(y)
  L <- .gp_chol(cov)
  diff <- y - mu
  alpha <- .gp_chol_solve(L, diff)
  quad <- sum(diff * alpha)
  logdet <- 2 * sum(log(diag(L)))
  -0.5 * (quad + logdet + n * log(2 * pi))
}

#' Synthetic log-likelihood
#'
#' Returns \code{log_lik, mu, cov} per Wilkinson eq. (2) and
#' Meeds & Welling eq. (9).
#' @param draws See Usage.
#' @param obs See Usage.
#' @param epsilon See Usage.
#' @param summary See Usage.
#' @export
synthetic_log_likelihood <- function(draws, obs, epsilon = 0, summary = NULL) {
  rows <- lapply(draws, function(z) .gp_summarise(z, summary))
  S <- length(rows)
  if (S < 2L) stop("synthetic_log_likelihood: need at least 2 simulations")
  J <- length(rows[[1]])
  y <- .gp_summarise(obs, summary)
  if (length(y) != J)
    stop("synthetic_log_likelihood: summary length mismatch")
  mu <- vapply(seq_len(J), function(j) sum(vapply(rows, function(r) r[j],
                                                   numeric(1))) / S,
               numeric(1))
  cov <- matrix(0, J, J)
  for (a in seq_len(J)) for (b in seq_len(J)) {
    cov[a, b] <- sum(vapply(rows, function(r) (r[a] - mu[a]) *
                              (r[b] - mu[b]), numeric(1))) / (S - 1L)
  }
  e2 <- as.numeric(epsilon) ^ 2
  for (j in seq_len(J)) cov[j, j] <- cov[j, j] + e2
  list(log_lik = .gp_mvn_logpdf(y, mu, cov), mu = mu, cov = cov)
}

#' .gp_corr
#'
#' A step of the abcgp_native implementation. Called by \code{.gp_profile_nll},
#' \code{gp_fit}, \code{gp_predict}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param a Numeric; combined arithmetically in the body.
#' @param b Numeric; combined arithmetically in the body.
#' @param lengthscale Numeric; combined arithmetically in the body.
#' @param kernel One of \code{"matern32"}, \code{"sqexp"}.
#' @return A numeric value.
#' @export
.gp_corr <- function(a, b, lengthscale, kernel) {
  r2 <- sum(((a - b) / lengthscale) ^ 2)
  if (kernel == "sqexp") return(exp(-0.5 * r2))
  r <- sqrt(r2)
  if (kernel == "matern32") {
    s <- sqrt(3) * r
    return((1 + s) * exp(-s))
  }
  s <- sqrt(5) * r
  (1 + s + s * s / 3) * exp(-s)
}

#' .gp_basis
#'
#' A step of the abcgp_native implementation. Called by \code{gp_predict}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param theta Numeric; combined arithmetically in the body.
#' @return A vector, from \code{c}.
#' @export
#' @examples
#' x <- c(1.2, 2.4, 3.1, 4.8, 5.3, 6.7, 7.1, 8.9)
#' res <- .gp_basis(theta = x)
#' res
.gp_basis <- function(theta) c(1, theta, theta * theta)

#' .gp_as_nugget
#'
#' A step of the abcgp_native implementation. Called by \code{gp_fit}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param nugget Optional; may be \code{NULL}. Coerced to numeric by the body, with
#' \code{as.numeric}.
#' @param n A count; the body uses it as \code{rep(...)}.
#' @return The value of \code{pmax}.
#' @export
.gp_as_nugget <- function(nugget, n) {
  if (is.null(nugget)) return(rep(1e-8, n))
  v <- as.numeric(nugget)
  if (length(v) == 1L) v <- rep(v, n)
  if (length(v) != n) stop("gp_fit: nugget length mismatch")
  pmax(v, 1e-12)
}

#' .gp_profile_nll
#'
#' A step of the abcgp_native implementation. Called by \code{.gp_mle_lengthscale}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param X A matrix; indexed by row and column.
#' @param y Numeric; combined arithmetically in the body.
#' @param ls Passed to \code{.gp_corr}.
#' @param nug A vector; indexed elementwise.
#' @param kernel Passed to \code{.gp_corr}.
#' @return A numeric value.
#' @export
.gp_profile_nll <- function(X, y, ls, nug, kernel) {
  n <- nrow(X)
  A <- matrix(0, n, n)
  for (i in seq_len(n)) for (j in seq_len(n)) {
    A[i, j] <- .gp_corr(X[i, ], X[j, ], ls, kernel) +
      if (i == j) nug[i] else 0
  }
  L <- tryCatch(.gp_chol(A), error = function(e) NULL)
  if (is.null(L)) return(Inf)
  H <- t(apply(X, 1, .gp_basis))
  if (is.matrix(H)) q <- ncol(H) else { q <- length(H)
  H <- t(H) }
  if (n <= q) return(Inf)
  Ainv_y <- .gp_chol_solve(L, y)
  Ainv_H <- matrix(0, n, q)
  for (k in seq_len(q)) Ainv_H[, k] <- .gp_chol_solve(L, H[, k])
  HtAinvH <- crossprod(H, Ainv_H)
  HtAinvy <- crossprod(H, Ainv_y)
  beta <- tryCatch(.gp_chol_solve(.gp_chol(HtAinvH), HtAinvy),
                   error = function(e) NULL)
  if (is.null(beta)) return(Inf)
  resid <- y - as.numeric(H %*% beta)
  Ainv_r <- .gp_chol_solve(L, resid)
  s2 <- sum(resid * Ainv_r) / (n - q)
  if (s2 <= 0) return(Inf)
  logdet <- 2 * sum(log(diag(L)))
  0.5 * (logdet + (n - q) * log(s2))
}

#' .gp_mle_lengthscale
#'
#' A step of the abcgp_native implementation. Called by \code{gp_fit}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param X A matrix; indexed by row and column.
#' @param y Passed to \code{.gp_profile_nll}.
#' @param nug Passed to \code{.gp_profile_nll}.
#' @param kernel Passed to \code{.gp_profile_nll}.
#' @return The value of \code{ls}, as built in the body.
#' @export
.gp_mle_lengthscale <- function(X, y, nug, kernel) {
  n <- nrow(X)
  p <- ncol(X)
  spans <- vapply(seq_len(p), function(j) {
    s <- max(X[, j]) - min(X[, j])
    if (s > 0) s else 1
  }, numeric(1))
  ls <- 0.5 * spans
  best <- .gp_profile_nll(X, y, ls, nug, kernel)
  grid <- c(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
  for (it in 1:3) {
    improved <- FALSE
    for (j in seq_len(p)) {
      for (g in grid) {
        trial <- ls
        trial[j] <- g * spans[j]
        val <- .gp_profile_nll(X, y, trial, nug, kernel)
        if (val < best - 1e-12) { best <- val
        ls <- trial
        improved <- TRUE }
      }
    }
    if (!improved) break
  }
  ls
}

#' Fit a GP emulator to a log-likelihood ensemble
#'
#' Wilkinson (2014) Sec. 2.1: quadratic mean, conjugate improper
#' \eqn{1/tau^2} prior on \eqn{(beta, tau^2)}, plug-in MLE length-scale.
#' @param design See Usage.
#' @param values See Usage.
#' @param nugget See Usage.
#' @param lengthscale See Usage.
#' @param kernel See Usage.
#' @param tau2 See Usage.
#' @export
gp_fit <- function(design, values, nugget = NULL, lengthscale = NULL,
                   kernel = "sqexp", tau2 = NULL) {
  if (!(kernel %in% .MORIE_GP_KERNELS))
    stop("gp_fit: kernel must be one of sqexp, matern32, matern52")
  X <- as.matrix(design)
  storage.mode(X) <- "double"
  y <- as.numeric(values)
  n <- nrow(X)
  if (n != length(y)) stop("gp_fit: design/values length mismatch")
  if (n < 3L) stop("gp_fit: need at least 3 design points")
  p <- ncol(X)
  if (is.null(lengthscale)) {
    nug_init <- .gp_as_nugget(nugget, n)
    ls <- .gp_mle_lengthscale(X, y, nug_init, kernel)
  } else {
    ls <- as.numeric(lengthscale)
    if (length(ls) == 1L) ls <- rep(ls, p)
  }
  if (any(ls <= 0)) stop("gp_fit: length-scales must be positive")
  nug <- .gp_as_nugget(nugget, n)
  A <- matrix(0, n, n)
  for (i in seq_len(n)) for (j in seq_len(n)) {
    A[i, j] <- .gp_corr(X[i, ], X[j, ], ls, kernel) +
      if (i == j) nug[i] else 0
  }
  L <- .gp_chol(A)
  H <- t(apply(X, 1, .gp_basis))
  if (is.matrix(H)) q <- ncol(H) else { q <- length(H)
  H <- t(H) }
  if (n <= q) stop("gp_fit: quadratic mean not identified by design")
  Ainv_y <- .gp_chol_solve(L, y)
  Ainv_H <- matrix(0, n, q)
  for (k in seq_len(q)) Ainv_H[, k] <- .gp_chol_solve(L, H[, k])
  HtAinvH <- crossprod(H, Ainv_H)
  HtAinvy <- crossprod(H, Ainv_y)
  Lh <- .gp_chol(HtAinvH)
  beta <- .gp_chol_solve(Lh, HtAinvy)
  resid <- y - as.numeric(H %*% beta)
  Ainv_r <- .gp_chol_solve(L, resid)
  if (is.null(tau2)) tau2 <- sum(resid * Ainv_r) / (n - q)
  list(design = X, values = y, beta = beta, tau2 = as.numeric(tau2),
       lengthscale = ls, kernel = kernel, nugget = nug, chol = L,
       Ainv_r = Ainv_r, Ainv_H = Ainv_H, H = H, HtAinvH_chol = Lh,
       n = n, q = q, dim = p)
}

#' Posterior mean and sd at theta
#' @param fit See Usage.
#' @param theta See Usage.
#' @export
gp_predict <- function(fit, theta) {
  X <- fit$design
  ls <- fit$lengthscale
  kern <- fit$kernel
  t <- as.numeric(theta)
  if (length(t) != fit$dim)
    stop("gp_predict: theta has wrong dimension")
  k <- vapply(seq_len(fit$n), function(i) .gp_corr(t, X[i, ], ls, kern),
              numeric(1))
  h <- .gp_basis(t)
  mean <- sum(h * fit$beta) + sum(k * fit$Ainv_r)
  Ainv_k <- .gp_chol_solve(fit$chol, k)
  var <- 1 - sum(k * Ainv_k)
  H <- fit$H
  hh <- vapply(seq_len(fit$q), function(j)
    h[j] - sum(H[, j] * Ainv_k), numeric(1))
  w <- .gp_chol_solve(fit$HtAinvH_chol, hh)
  var <- var + sum(hh * w)
  var <- fit$tau2 * max(var, 0)
  c(mean = as.numeric(mean), sd = sqrt(var))
}

#' Wilkinson implausibility rule, eq. (3)
#' @param fit See Usage.
#' @param theta See Usage.
#' @param threshold See Usage.
#' @param n_sd See Usage.
#' @export
implausible <- function(fit, theta, threshold = 10, n_sd = 3) {
  pr <- gp_predict(fit, theta)
  as.logical(pr["mean"] + n_sd * pr["sd"] <
               max(fit$values) - threshold)
}

#' Sequential history matching
#'
#' Wilkinson (2014) Sec. 3: waves of design, emulate, rule out,
#' redesign. Returns the GP fit and a list of per-wave summaries.
#' @param sim See Usage.
#' @param obs See Usage.
#' @param prior_ppf See Usage.
#' @param n_waves See Usage.
#' @param n_design See Usage.
#' @param n_sim See Usage.
#' @param epsilon See Usage.
#' @param summary See Usage.
#' @param threshold See Usage.
#' @param n_sd See Usage.
#' @param kernel See Usage.
#' @param accept_kernel See Usage.
#' @param seed See Usage.
#' @export
history_match <- function(sim, obs, prior_ppf, n_waves = 3L, n_design = 32L,
                          n_sim = 50L, epsilon = 1, summary = NULL,
                          threshold = 10, n_sd = 3, kernel = "sqexp",
                          accept_kernel = "gaussian", seed = 0L) {
  ensemble_x <- list()
  ensemble_y <- numeric(0)
  ensemble_v <- numeric(0)
  waves <- list()
  fit <- NULL
  for (w in seq_len(as.integer(n_waves))) {
    cand <- design_from_prior(as.integer(n_design) * 4L, prior_ppf,
                              skip = 1L + (w - 1L) *
                                as.integer(n_design) * 4L)
    rows <- lapply(seq_len(nrow(cand)), function(i) as.numeric(cand[i, ]))
    if (!is.null(fit)) {
      keep_idx <- vapply(rows, function(r) !implausible(fit, r, threshold, n_sd),
                         logical(1))
      ruled <- sum(!keep_idx)
      rows <- rows[keep_idx]
      if (length(rows) == 0L) rows <- rows[seq_len(min(length(rows),
                                                      as.integer(n_design)))]
    } else {
      ruled <- 0L
    }
    if (length(rows) > as.integer(n_design)) rows <- rows[seq_len(as.integer(n_design))]
    for (i in seq_along(rows)) {
      ll <- gabc_log_likelihood(sim, obs, rows[[i]], n_sim = n_sim,
                                epsilon = epsilon, summary = summary,
                                kernel = accept_kernel,
                                seed = as.integer(seed) + 1000L * (w - 1L) +
                                  (i - 1L))
      if (is.finite(ll$log_lik)) {
        ensemble_x[[length(ensemble_x) + 1L]] <- rows[[i]]
        ensemble_y <- c(ensemble_y, ll$log_lik)
        ensemble_v <- c(ensemble_v, ll$nugget_variance)
      }
    }
    if (length(ensemble_x) < 3L)
      stop("history_match: wave left fewer than 3 usable points")
    X <- do.call(rbind, ensemble_x)
    fit <- gp_fit(X, ensemble_y, nugget = ensemble_v, kernel = kernel)
    waves[[length(waves) + 1L]] <- list(wave = w - 1L,
                                        n_ensemble = length(ensemble_x),
                                        ruled_implausible = ruled,
                                        max_log_lik = max(ensemble_y))
  }
  list(fit = fit, waves = waves)
}

#' .gp_alpha_terms
#'
#' A step of the abcgp_native implementation. Called by \code{.gp_mw_sampler}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param log_prior Accepted by the signature and not used anywhere in the body.
#' @param theta Passed to \code{log_prior}.
#' @param theta_p Passed to \code{log_prior}.
#' @param ll Numeric; combined arithmetically in the body.
#' @param ll_p Numeric; combined arithmetically in the body.
#' @param log_q Numeric; combined arithmetically in the body.
#' @param log_q_p Numeric; combined arithmetically in the body.
#' @return A numeric value.
#' @export
.gp_alpha_terms <- function(log_prior, theta, theta_p, ll, ll_p,
                            log_q, log_q_p) {
  min(0, (log_prior(theta_p) + ll_p + log_q_p) -
        (log_prior(theta) + ll + log_q))
}

#' .gp_expected_error
#'
#' A step of the abcgp_native implementation. Called by \code{.gp_mw_sampler}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param alphas A vector; its length is taken.
#' @param tau Passed to \code{<=}.
#' @param n_grid A count; the body uses it as \code{seq_len(...)}. Defaults to \code{101L}.
#' @return A numeric value.
#' @export
.gp_expected_error <- function(alphas, tau, n_grid = 101L) {
  M <- length(alphas)
  total <- 0
  for (i in seq_len(n_grid)) {
    u <- (i - 0.5) / n_grid
    if (u <= tau) err <- sum(alphas < u) / M
    else err <- sum(alphas >= u) / M
    total <- total + err
  }
  total / n_grid
}

#' .gp_median
#'
#' A step of the abcgp_native implementation. Called by \code{.gp_mw_sampler}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param v Numeric; passed to \code{sort}.
#' @return One of two values, depending on the branch taken.
#' @export
#' @examples
#' x <- c(1.2, 2.4, 3.1, 4.8, 5.3, 6.7, 7.1, 8.9)
#' res <- .gp_median(v = x)
#' res
.gp_median <- function(v) {
  s <- sort(v)
  n <- length(s)
  if (n %% 2L == 1L) s[(n + 1L) %/% 2L] else 0.5 * (s[n %/% 2L] + s[n %/% 2L + 1L])
}

#' .gp_draw_mean
#'
#' A step of the abcgp_native implementation. Called by \code{.gp_mw_sampler}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param mu A vector; its length is taken.
#' @param cov Numeric; combined arithmetically in the body.
#' @param S Numeric; combined arithmetically in the body.
#' @param e Passed to \code{.ghc_norm}.
#' @return A numeric value.
#' @export
.gp_draw_mean <- function(mu, cov, S, e) {
  n <- length(mu)
  scaled <- cov / S
  L <- .gp_chol(scaled)
  z <- .ghc_norm(e, n)
  mu + L %*% z
}

#' .gp_mw_sampler
#'
#' A step of the abcgp_native implementation. Called by \code{gps_abc}, \code{synthetic_abc}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param sim Accepted by the signature and not used anywhere in the body.
#' @param obs Passed to \code{.gp_summarise}.
#' @param log_prior Passed to \code{.gp_alpha_terms}.
#' @param theta0 Coerced to numeric by the body, with \code{as.numeric}.
#' @param n_iter Coerced to integer by the body, with \code{as.integer}.
#' @param n_sim Coerced to integer by the body, with \code{as.integer}.
#' @param epsilon Passed to \code{synthetic_log_likelihood}.
#' @param proposal_sd Coerced to numeric by the body, with \code{as.numeric}.
#' @param summary Passed to \code{.gp_summarise}.
#' @param seed Passed to \code{.ghc_rng}.
#' @param adaptive A flag; the body branches on it.
#' @param xi Passed to \code{<}.
#' @param delta_s Coerced to integer by the body, with \code{as.integer}.
#' @param n_alpha A count; the body uses it as \code{seq_len(...)}.
#' @param max_sim Optional; may be \code{NULL}. Coerced to integer by the body, with
#' \code{as.integer}.
#' @return A list with \code{chain}, \code{acceptance_rate}, \code{n_simulations},
#' \code{unresolved_steps}.
#' @export
.gp_mw_sampler <- function(sim, obs, log_prior, theta0, n_iter, n_sim,
                           epsilon, proposal_sd, summary, seed, adaptive,
                           xi, delta_s, n_alpha, max_sim = NULL) {
  e <- .ghc_rng(seed)
  theta <- as.numeric(theta0)
  p <- length(theta)
  sd <- as.numeric(proposal_sd)
  if (length(sd) == 1L) sd <- rep(sd, p)
  chain <- matrix(0, as.integer(n_iter) + 1L, p)
  chain[1, ] <- theta
  n_accept <- 0L
  unresolved <- 0L
  sims_used <- 0L
  for (it in seq_len(as.integer(n_iter))) {
    prop <- theta + sd * .ghc_norm(e, p)
    if (log_prior(prop) == -Inf) { chain[it + 1L, ] <- theta
    next }
    S <- as.integer(n_sim)
    repeat {
      cur <- lapply(seq_len(S), function(k) sim(theta, e))
      new <- lapply(seq_len(S), function(k) sim(prop, e))
      sims_used <- sims_used + 2L * S
      sl_c <- synthetic_log_likelihood(cur, obs, epsilon = epsilon,
                                       summary = summary)
      sl_p <- synthetic_log_likelihood(new, obs, epsilon = epsilon,
                                       summary = summary)
      ll_c <- sl_c$log_lik
      ll_p <- sl_p$log_lik
      if (!adaptive) {
        loga <- .gp_alpha_terms(log_prior, theta, prop, ll_c, ll_p, 0, 0)
        tau <- exp(loga)
        break
      }
      y <- .gp_summarise(obs, summary)
      alphas <- numeric(n_alpha)
      for (a in seq_len(n_alpha)) {
        mc <- .gp_draw_mean(sl_c$mu, sl_c$cov, S, e)
        mp <- .gp_draw_mean(sl_p$mu, sl_p$cov, S, e)
        loga <- min(0, (log_prior(prop) + .gp_mvn_logpdf(y, mp, sl_p$cov)) -
                      (log_prior(theta) + .gp_mvn_logpdf(y, mc, sl_c$cov)))
        alphas[a] <- exp(loga)
      }
      tau <- .gp_median(alphas)
      err <- .gp_expected_error(alphas, tau)
      if (err < xi) break
      if (!is.null(max_sim) && S >= as.integer(max_sim)) { unresolved <- unresolved + 1L
      break }
      S <- S + as.integer(delta_s)
    }
    if (.ghc_unif(e, 1L) <= tau) { theta <- prop
    n_accept <- n_accept + 1L }
    chain[it + 1L, ] <- theta
  }
  list(chain = chain, acceptance_rate = n_accept / as.integer(n_iter),
       n_simulations = sims_used, unresolved_steps = unresolved)
}

#' Synthetic-likelihood ABC, Meeds & Welling Algorithm 1
#' @param sim See Usage.
#' @param obs See Usage.
#' @param log_prior See Usage.
#' @param theta0 See Usage.
#' @param n_iter See Usage.
#' @param n_sim See Usage.
#' @param epsilon See Usage.
#' @param proposal_sd See Usage.
#' @param summary See Usage.
#' @param seed See Usage.
#' @export
synthetic_abc <- function(sim, obs, log_prior, theta0, n_iter = 200L,
                          n_sim = 20L, epsilon = 0, proposal_sd = 0.5,
                          summary = NULL, seed = 0L) {
  .gp_mw_sampler(sim, obs, log_prior, theta0, n_iter, n_sim, epsilon,
                 proposal_sd, summary, seed, adaptive = FALSE, xi = NULL,
                 delta_s = 0L, n_alpha = 0L, max_sim = NULL)
}

#' GPS-ABC adaptive sampler, Meeds & Welling Algorithm 2
#' @param sim See Usage.
#' @param obs See Usage.
#' @param log_prior See Usage.
#' @param theta0 See Usage.
#' @param n_iter See Usage.
#' @param n_sim See Usage.
#' @param epsilon See Usage.
#' @param proposal_sd See Usage.
#' @param summary See Usage.
#' @param seed See Usage.
#' @param xi See Usage.
#' @param delta_s See Usage.
#' @param n_alpha See Usage.
#' @param max_sim See Usage.
#' @export
gps_abc <- function(sim, obs, log_prior, theta0, n_iter = 200L, n_sim = 10L,
                    epsilon = 0, proposal_sd = 0.5, summary = NULL,
                    seed = 0L, xi = 0.05, delta_s = 10L, n_alpha = 64L,
                    max_sim = 400L) {
  .gp_mw_sampler(sim, obs, log_prior, theta0, n_iter, n_sim, epsilon,
                 proposal_sd, summary, seed, adaptive = TRUE, xi = xi,
                 delta_s = delta_s, n_alpha = n_alpha, max_sim = max_sim)
}

#' ABC with a GP surrogate, four methods
#'
#' Wilkinson (2014) history matching + GP-emulated log-posterior on a
#' grid, or one of the Meeds & Welling (2014) samplers.
#' @param sim See Usage.
#' @param obs See Usage.
#' @param X_grid See Usage.
#' @param kernel See Usage.
#' @param method See Usage.
#' @param prior_ppf See Usage.
#' @param log_prior See Usage.
#' @param theta0 See Usage.
#' @param n_sim See Usage.
#' @param epsilon See Usage.
#' @param summary See Usage.
#' @param n_waves See Usage.
#' @param n_design See Usage.
#' @param threshold See Usage.
#' @param n_sd See Usage.
#' @param accept_kernel See Usage.
#' @param n_iter See Usage.
#' @param proposal_sd See Usage.
#' @param xi See Usage.
#' @param delta_s See Usage.
#' @param n_alpha See Usage.
#' @param seed See Usage.
#' @export
abc_gp_emulator <- function(sim, obs, X_grid = NULL, kernel = "sqexp",
                            method = "wilkinson", prior_ppf = NULL,
                            log_prior = NULL, theta0 = NULL, n_sim = 50L,
                            epsilon = 1.0, summary = NULL, n_waves = 3L,
                            n_design = 32L, threshold = 10, n_sd = 3,
                            accept_kernel = "gaussian", n_iter = 200L,
                            proposal_sd = 0.5, xi = 0.05, delta_s = 10L,
                            n_alpha = 64L, seed = 0L) {
  if (!(method %in% .MORIE_GP_METHODS))
    stop("abc_gp_emulator: method must be one of wilkinson, gps, adaptive, synthetic")
  if (!is.function(sim)) stop("abc_gp_emulator: sim must be a callable")
  if (method == "wilkinson") {
    if (is.null(prior_ppf))
      stop("abc_gp_emulator: method='wilkinson' needs prior_ppf")
    hm <- history_match(sim, obs, prior_ppf, n_waves = n_waves,
                        n_design = n_design, n_sim = n_sim, epsilon = epsilon,
                        summary = summary, threshold = threshold, n_sd = n_sd,
                        kernel = kernel, accept_kernel = accept_kernel,
                        seed = seed)
    fit <- hm$fit
    if (is.null(X_grid)) grid <- fit$design
    else grid <- as.matrix(X_grid)
    storage.mode(grid) <- "double"
    means <- sapply(seq_len(nrow(grid)), function(i) gp_predict(fit, grid[i, ])["mean"])
    sds <- sapply(seq_len(nrow(grid)), function(i) gp_predict(fit, grid[i, ])["sd"])
    top <- which.max(means)
    lse <- .gp_lse(means)
    post <- exp(means - lse)
    return(list(estimate = unname(grid[top, ]), grid = grid,
                log_likelihood = as.numeric(means),
                log_likelihood_sd = as.numeric(sds),
                posterior = as.numeric(post), waves = hm$waves,
                ensemble_size = fit$n, lengthscale = fit$lengthscale,
                tau2 = fit$tau2, beta = fit$beta,
                n_simulations = sum(vapply(hm$waves,
                                           function(w) w$n_ensemble,
                                           integer(1))) * as.integer(n_sim),
                method = "ABC GP emulator, Wilkinson (2014) with sequential history matching"))
  }
  if (is.null(log_prior) || is.null(theta0))
    stop("abc_gp_emulator: sampler methods need log_prior and theta0")
  if (method == "synthetic") {
    out <- synthetic_abc(sim, obs, log_prior, theta0, n_iter = n_iter,
                         n_sim = n_sim, epsilon = epsilon,
                         proposal_sd = proposal_sd, summary = summary,
                         seed = seed)
    label <- "synthetic-likelihood ABC-MH, Meeds & Welling Algorithm 1"
  } else {
    out <- gps_abc(sim, obs, log_prior, theta0, n_iter = n_iter,
                   n_sim = n_sim, epsilon = epsilon,
                   proposal_sd = proposal_sd, summary = summary, seed = seed,
                   xi = xi, delta_s = delta_s, n_alpha = n_alpha)
    label <- "GPS-ABC adaptive MH, Meeds & Welling (2014) Algorithm 2, eqs. 11-16"
  }
  chain <- out$chain
  p <- ncol(chain)
  burn <- nrow(chain) %/% 2L
  kept <- chain[(burn + 1L):nrow(chain), , drop = FALSE]
  est <- colMeans(kept)
  out$estimate <- est
  out$posterior_mean <- est
  out$burn_in <- burn
  out$method <- label
  out
}

#' Compact alias for abc_gp_emulator, per ledger/NAMING.md
#' @export
#' @noRd
abcgpemulator <- abc_gp_emulator

# house entry point: the package exports one morie_<module>
morie_abcgp <- abc_gp_emulator
