# Sequential Monte Carlo: the bootstrap particle filter.
# Sources: King, A. A., Nguyen, D. & Ionides, E. L. (2016) "Statistical
# Inference for Partially Observed Markov Processes: The R Package
# pomp", J. Statistical Software 69(12), 1-43, doi:10.18637/jss.v069.i12
# (Algorithm 1: SMC, Algorithm 2: systematic resampling);
# Gordon, N. J., Salmond, D. J. & Smith, A. F. M. (1993) "Novel approach
# to nonlinear/non-Gaussian Bayesian state estimation", IEE Proc. F 140(2),
# 107-113, doi:10.1049/ip-f-2.1993.0015; Doucet, A. & Johansen, A. M.
# (2011) in The Oxford Handbook of Nonlinear Filtering; Kalman, R. E.
# (1960) for the linear-Gaussian check.
#
# Native implementation mirroring morie.fn.prtcl exactly: Algorithm 1
# (propagate, weight, resample), Algorithm 2 (one uniform, J evenly
# spaced points) for systematic resampling, the same ESS formula, and
# the same log of the unbiased one-step predictive density (downward
# Jensen bias, not corrected here).

#' Effective sample size
#'
#' \code{(sum w)^2 / sum w^2}: how many particles are really
#' contributing.
#'
#' @param weights Numeric vector of positive weights.
#' @return Numeric scalar.
#' @export
effective_sample_size <- function(weights) {
  s1 <- sum(weights)
  s2 <- sum(weights^2)
  if (s2 <= 0) return(0)
  s1 * s1 / s2
}

#' Systematic resampling
#'
#' One uniform, J evenly spaced points through the cumulative weights;
#' the count particle j receives differs from \code{J * w_j} by less
#' than one, deterministically.
#'
#' @param weights Numeric vector of positive weights.
#' @param u Optional fixed offset in \code{[0, 1)}; if \code{NULL} one
#'   uniform is drawn from the shared generator.
#' @return Integer vector of indices.
#' @export
systematic_resample <- function(weights, u = NULL) {
  J <- length(weights)
  tot <- sum(weights)
  if (tot <= 0)
    stop("prtcl: all particle weights are zero; the filter has lost ",
         "the signal")
  w <- as.numeric(weights) / tot
  if (is.null(u)) u <- .ghc_unif(.ghc_rng(0L), 1L)
  if (u < 0 || u >= 1)
    stop(sprintf("prtcl: the offset must lie in [0, 1), got %r", u))
  idx <- integer(J)
  cum_ <- w[1L]
  j <- 1L
  for (m in seq_len(J)) {
    pos <- (m - 1L + u) / J
    while (pos > cum_ && j < J) {
      j <- j + 1L
      cum_ <- cum_ + w[j]
    }
    idx[m] <- j
  }
  idx
}

# Internal multinomial resampler used when systematic=FALSE
#' Internal multinomial resampler used when systematic=FALSE
#'
#' A step of the prtcl_native implementation. Called by \code{morie_prtcl}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param weights A vector; its length is taken.
#' @return The value of \code{idx}, as built in the body.
#' @export
#' @examples
#' x <- c(1.2, 2.4, 3.1, 4.8, 5.3, 6.7, 7.1, 8.9)
#' res <- .multinomial_resample(weights = x)
#' res
.multinomial_resample <- function(weights) {
  J <- length(weights)
  w <- as.numeric(weights) / sum(weights)
  e <- .ghc_rng(0L)
  u <- .ghc_unif(e, J)
  ord <- order(u)
  rU <- u[ord]
  cw <- cumsum(w)
  cw[J] <- 1.0
  idx <- rep(0L, J)
  i <- 1L
  for (k in seq_len(J)) {
    while (i < J && cw[i] < rU[k]) i <- i + 1L
    idx[ord[k]] <- i
  }
  idx
}

#' .scalar
#'
#' A step of the prtcl_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param state A vector; indexed elementwise.
#' @return A vector, from \code{as.numeric}.
#' @export
.scalar <- function(state) {
  if (is.list(state)) return(as.numeric(state[[1L]]))
  as.numeric(state)
}

#' Bootstrap particle filter (Algorithm 1 of King, Nguyen & Ionides 2016)
#'
#' Propagate each particle, weight by the measurement density, resample
#' when the ESS falls below \code{resample_threshold * J}.
#'
#' @param y Numeric vector of observations.
#' @param n_particles Integer, number of particles.
#' @param init Function \code{(rng) -> state} for one particle.
#' @param step Function \code{(state, t, rng) -> state} for the process.
#' @param loglik Function \code{(state, obs, t) -> float} for the
#'   measurement density.
#' @param seed Seed for the shared generator.
#' @param resample_threshold Resample when ESS drops below this
#'   fraction of \code{n_particles}; default 1.0 (every step).
#' @param systematic If \code{TRUE} use systematic resampling
#'   (Algorithm 2); otherwise multinomial.
#' @return A list with \code{estimate}, \code{filtered_mean},
#'   \code{loglik}, \code{ess}, \code{min_ess}, \code{resampled},
#'   \code{n_particles}, \code{n_obs}, \code{systematic},
#'   \code{particles}, \code{method}.
#' @export
morie_prtcl <- function(y, n_particles, init, step, loglik, seed = 0L,
                        resample_threshold = 1.0,
                        systematic = TRUE) {
  obs <- as.numeric(y)
  N <- length(obs)
  J <- as.integer(n_particles)
  if (J < 2L)
    stop(sprintf("prtcl: need at least 2 particles, got %d", J))
  if (N == 0L) stop("prtcl: no observations")
  if (resample_threshold <= 0 || resample_threshold > 1)
    stop(sprintf("prtcl: resample_threshold must be in (0, 1], got %r",
                 resample_threshold))
  e <- .ghc_rng(seed)
  parts <- lapply(seq_len(J), function(i) init(e))
  ll <- 0.0
  means <- numeric(N)
  esss <- numeric(N)
  resampled <- logical(N)
  for (n in seq_len(N)) {
    parts <- lapply(seq_along(parts), function(j) step(parts[[j]], n - 1L, e))
    lw <- vapply(seq_along(parts), function(j)
      loglik(parts[[j]], obs[n], n - 1L), numeric(1))
    mx <- max(lw)
    if (mx == -Inf)
      stop(sprintf("prtcl: every particle has zero likelihood at ",
                   "observation %d", n - 1L))
    w <- exp(lw - mx)
    tot <- sum(w)
    ll <- ll + mx + log(tot / J)
    ess <- effective_sample_size(w)
    esss[n] <- ess
    means[n] <- sum(w * vapply(parts, .scalar, numeric(1))) / tot
    if (ess < resample_threshold * J) {
      idx <- if (systematic) systematic_resample(w)
             else .multinomial_resample(w)
      parts <- parts[idx]
      resampled[n] <- TRUE
    } else {
      resampled[n] <- FALSE
    }
  }
  list(estimate = means, filtered_mean = means, loglik = ll,
       ess = esss, min_ess = min(esss), resampled = resampled,
       n_particles = J, n_obs = N, systematic = isTRUE(systematic),
       particles = parts,
       method = paste0("bootstrap particle filter, ",
                       "King, Nguyen & Ionides (2016) Algorithm 1 ",
                       "with systematic resampling (Algorithm 2)"))
}

#' Kalman filter for the scalar linear-Gaussian state-space model
#'
#' \code{x_n = a x_{n-1} + N(0, q)}, \code{y_n = c x_n + N(0, r)},
#' starting at \code{(m0, p0)}. Provided so the particle filter can
#' be checked against the exact answer.
#'
#' @param y Numeric vector of observations.
#' @param a Process coefficient.
#' @param q Process variance.
#' @param c Measurement coefficient.
#' @param r Measurement variance.
#' @param m0 Initial state mean.
#' @param p0 Initial state variance.
#' @return A list with \code{filtered_mean} and \code{loglik}.
#' @export
kalman_filter_1d <- function(y, a, q, c, r, m0 = 0.0, p0 = 1.0) {
  m <- as.numeric(m0)
  p <- as.numeric(p0)
  means <- numeric(length(y))
  ll <- 0.0
  for (i in seq_along(y)) {
    m <- a * m
    p <- a * a * p + q
    s <- c * c * p + r
    v <- y[i] - c * m
    ll <- ll - 0.5 * (log(2 * pi * s) + v * v / s)
    gain <- p * c / s
    m <- m + gain * v
    p <- (1.0 - gain * c) * p
    means[i] <- m
  }
  list(filtered_mean = means, loglik = ll)
}

#' @export
particlefilter <- morie_prtcl

#' @export
prtcl_cheatsheet <- function() {
  paste0("prtcl: propagate, weight by the measurement density, ",
         "resample (pomp Alg. 1). Mean weight per step gives an ",
         "UNBIASED likelihood -- so its LOG is biased DOWNWARD by ",
         "Jensen, and comparing models at different particle counts ",
         "compares the counts. Systematic resampling (Alg. 2) gives ",
         "each particle a count within 1 of J*w_j deterministically. ",
         "Watch ESS: a degenerate filter still returns numbers.")
}
