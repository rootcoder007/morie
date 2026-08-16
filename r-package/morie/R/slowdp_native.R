# Truncating a Dirichlet process, and knowing what you dropped.
# Sources: Sethuraman, J. (1994) "A Constructive Definition of
# Dirichlet Priors", Statistica Sinica 4(2), 639-650 (the stick-
# breaking construction V_k ~ Beta(1, alpha) and the closed-form
# expected tail (alpha/(1+alpha))^K); Ferguson, T. S. (1973) Ann.
# Statist. 1(2), 209-230 (the DP itself); Ishwaran, H. & James,
# L. F. (2001) JASA 96(453), 161-173 (truncated stick-breaking);
# Neal, R. M. (2000) JCGS 9(2), 249-265 (the sampling context).
#
# Native R arm mirroring the Python arm exactly: the same V_k ~
# Beta(1, alpha) draws from the shared generator through inverse
# transform, the same closed-form expected tail and its inversion for
# choosing K, the same diagnostics reporting the realised tail
# against the expected one, and the same renormalisation step that
# silently moves the discarded mass onto the survivors and so has
# its effect reported alongside.

.slowdp_EPS <- 1e-12

#' .beta_1_alpha
#'
#' A step of the slowdp_native implementation. Called by \code{stick_breaking}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param e Passed to \code{.ghc_unif}.
#' @param alpha Numeric; combined arithmetically in the body.
#' @return A numeric value.
#' @export
.beta_1_alpha <- function(e, alpha) {
  u <- .ghc_unif(e, 1L)
  u <- min(max(u, 1e-15), 1 - 1e-15)
  1 - u^(1 / alpha)
}

#' Sethuraman stick-breaking weights
#' @export
stick_breaking <- function(alpha, K, seed = 0) {
  a <- as.numeric(alpha)
  n <- as.integer(K)
  if (a <= 0) stop("slowdp: alpha must be positive")
  if (n < 1L) stop("slowdp: at least one stick is needed")
  e <- .ghc_rng(seed)
  p <- numeric(n); Vs <- numeric(n); rest <- 1
  for (k in seq_len(n)) {
    v <- .beta_1_alpha(e, a)
    Vs[k] <- v
    p[k] <- v * rest
    rest <- rest * (1 - v)
  }
  list(weights = p, V = Vs, remaining = rest,
       kept_mass = sum(p), K = n, alpha = a,
       note = paste("the remaining stick is the mass truncation ",
                    "throws away"))
}

#' Closed-form expected tail (alpha/(1+alpha))^K
#' @export
truncation_error <- function(alpha, K) {
  a <- as.numeric(alpha)
  n <- as.integer(K)
  if (a <= 0 || n < 1L)
    stop("slowdp: need alpha > 0 and K >= 1")
  e <- (a / (1 + a))^n
  list(expected_tail = e, kept = 1 - e, alpha = a, K = n,
       per_stick_factor = a / (1 + a),
       note = paste("a more diffuse process needs more sticks for ",
                    "the same fidelity"))
}

#' Smallest K with expected tail below the tolerance
#' @export
sticks_for_tolerance <- function(alpha, tol = 1e-3) {
  a <- as.numeric(alpha)
  t <- as.numeric(tol)
  if (a <= 0) stop("slowdp: alpha must be positive")
  if (!(t > 0 && t < 1)) {
    stop("slowdp: the tolerance must lie in (0,1)")
  }
  f <- a / (1 + a)
  K <- as.integer(ceiling(log(t) / log(f)))
  K <- max(1L, K)
  list(K = K, expected_tail = truncation_error(a, K)$expected_tail,
       tolerance = t,
       note = paste("chosen from the closed form, not guessed"))
}

#' Realised tail vs the expected geometric tail
#' @export
decay_diagnostics <- function(weights, alpha) {
  p <- as.numeric(weights)
  K <- length(p)
  if (K < 1L) stop("slowdp: no weights given")
  exp_tail <- truncation_error(alpha, K)$expected_tail
  realised <- max(0, 1 - sum(p))
  biggest_late <- which.max(p) - 1L
  monotone <- all(diff(p) >= -.slowdp_EPS)
  list(realised_tail = realised, expected_tail = exp_tail,
       ratio = if (exp_tail > .slowdp_EPS) realised / exp_tail else Inf,
       largest_index = biggest_late, monotone = monotone,
       note = paste("the sticks are NOT ordered; a late large stick ",
                    "is exactly what a mean-based truncation misses"))
}

#' Truncated Dirichlet process draw
#' @export
truncated_dp <- function(alpha, K, base_sampler = NULL, seed = 0,
                         renormalise = TRUE) {
  e <- .ghc_rng(seed)
  sb <- stick_breaking(alpha, K, seed = seed)
  p <- sb$weights
  tail <- sb$remaining
  if (isTRUE(renormalise)) {
    z <- sum(p)
    if (z <= .slowdp_EPS) {
      stop("slowdp: the kept sticks carry no mass")
    }
    p <- p / z
  }
  atoms <- if (!is.null(base_sampler)) {
    vapply(seq_len(as.integer(K)),
           function(i) as.numeric(base_sampler(e)),
           numeric(1))
  } else seq_len(as.integer(K)) - 1L
  list(estimate = p, weights = p, atoms = atoms,
       discarded_mass = tail,
       expected_discarded = truncation_error(alpha, K)$expected_tail,
       renormalised = isTRUE(renormalise), K = as.integer(K),
       alpha = as.numeric(alpha),
       method = "truncated stick-breaking; Sethuraman (1994)",
       note = paste("renormalising moves the discarded mass onto ",
                    "the survivors, which is why the amount is ",
                    "returned"))
}

# house entry point: the package exports one morie_<module>
morie_slowdp <- truncated_dp
