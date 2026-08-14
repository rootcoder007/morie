# smcopt_native.R -- mirror of smcopt_python_reference.py

# SMC for global optimisation (annealed sequence of targets).
# Sources: Del Moral, P., Doucet, A. & Jasra, A. (2006) "Sequential
# Monte Carlo samplers", *JRSS-B* 68(3), 411-436, section 2.3.1(c).
#
# Native implementation mirroring Python morie.fn.smcopt exactly:
# the same increasing phi_n ladder in either geometric or linear
# flavour, the same smcsam front end with the random-walk kernel
# whose scale shrinks as phi grows, the same "best point seen"
# tracking that the resampling step does NOT overwrite, and the
# same payload keys.

annealing_ladder <- function(n_steps, phi_max = 50.0, phi_min = 0.1,
                             kind = "geometric") {
  n <- as.integer(n_steps)
  if (n < 2L)
    stop("smcopt: need at least two steps, got ", n)
  if (phi_min <= 0 || phi_max <= phi_min)
    stop("smcopt: need 0 < phi_min < phi_max")
  if (kind == "geometric") {
    r <- (phi_max / phi_min) ^ (1 / (n - 1))
    return(phi_min * r ^ (seq_len(n) - 1L))
  }
  if (kind == "linear")
    return(phi_min + (phi_max - phi_min) * (seq_len(n) - 1L) / (n - 1L))
  stop("smcopt: kind must be 'geometric' or 'linear', got ",
       deparse(kind))
}

# Bare random-walk kernel with a caller-chosen scale.  Mirrors
# the Python arm's random_walk_kernel(scale=1.0): propose
# x' = x + Normal(0, scale^2) and accept with the standard
# Metropolis-Hastings ratio.  No library calls; uses .ghc_norm
# so the run is reproducible from the seed.
.smcopt_rwk <- function(scale) {
  function(x, log_target, rng) {
    n <- length(x)
    prop <- as.numeric(x) + scale * .ghc_norm(rng, n)
    lp <- log_target(prop)
    if (!is.finite(lp))
      return(list(x = x, accept = 0L))
    cur <- log_target(x)
    log_alpha <- lp - cur
    if (log(runif(1, 0, 1)) < log_alpha)
      return(list(x = prop, accept = 1L))
    list(x = x, accept = 0L)
  }
}

smcopt <- function(objective, initial, n_particles = 200L, n_steps = 30L,
                   phi_max = 50.0, phi_min = 0.1, kind = "geometric",
                   kernel = NULL, ess_threshold = 0.5, scheme = "systematic",
                   seed = 0L, maximise = TRUE) {
  sign <- if (isTRUE(maximise)) 1.0 else -1.0
  ladder <- annealing_ladder(n_steps, phi_max, phi_min, kind)
  best_x <- NULL
  best_v <- -Inf

  log_gamma <- function(x, phi) {
    v <- sign * as.numeric(objective(x))
    if (v > best_v) {
      best_v <<- v
      best_x <<- as.numeric(x)
    }
    phi * v
  }

  if (is.null(kernel)) {
    base <- .smcopt_rwk(1.0)
    kern <- function(x, log_target, rng) base(x, log_target, rng)
  } else {
    kern <- kernel
  }

  fit <- smcsam(log_gamma, initial, n_particles = n_particles,
                ladder = ladder, kernel = kern,
                ess_threshold = ess_threshold, scheme = scheme,
                seed = seed)
  if (is.null(best_x))
    stop("smcopt: the objective was never evaluated")
  list(estimate = best_x,
       best_x = best_x,
       best_value = sign * best_v,
       particles = fit$particles,
       weights = fit$weights,
       particle_mean = fit$mean,
       ladder = ladder,
       ess_trace = fit$ess_trace,
       resampled = fit$resampled,
       accept_trace = fit$accept_trace,
       n_particles = as.integer(n_particles),
       maximise = isTRUE(maximise),
       note = paste("annealing concentrates on the modes but cannot ",
                    "find one no particle visits; widen `initial` ",
                    "before raising phi_max", sep = ""),
       method = paste("annealed SMC optimisation (Del Moral, Doucet ",
                      "& Jasra 2006, section 2.3.1c)", sep = ""))
}

.smcopt_cheatsheet <- function() {
  paste("smcopt: SMC as a global optimiser (Del Moral, Doucet & Jasra ",
        "2006, sec 2.3.1c). Anneal pi_n = pi^phi_n with phi rising, so ",
        "the target concentrates on the modes. Unlike single-chain ",
        "simulated annealing the particles INTERACT: resampling kills ",
        "the ones in poor modes and copies the ones in good modes. ",
        "Shares the sampler, weights and resampling with smcsam.",
        sep = "")
}

smc_optimise <- smcopt
sequential_mc <- smcopt
sequentialmc <- smcopt

# Native entry point.
morie_smcopt <- smcopt
