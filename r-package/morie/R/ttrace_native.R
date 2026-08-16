# Contact tracing and isolation: when is it enough on its own?
# Sources: Hellewell, J., Abbott, S., Gimma, A., Bosse, N. I., Jarvis,
# C. I., Russell, T. W., Munday, J. D., Kucharski, A. J., Edmunds, W.
# J., Centre for the Mathematical Modelling of Infectious Diseases
# COVID-19 Working Group, Funk, S. & Eggo, R. M. (2020) "Feasibility
# of controlling COVID-19 outbreaks by isolation of cases and
# contacts", The Lancet Global Health 8, e488-e496. Section "Methods
# -- Model structure": the negative binomial offspring distribution,
# serial-interval assignment, the rule that secondary cases arise only
# before the infector's isolation, initial outbreak sizes of 5/20/40,
# isolation assumed 100% effective, and the 100%/90% symptomatic
# split. Lloyd-Smith, J. O., Schreiber, S. J., Kopp, P. E. & Getz, W.
# M. (2005) "Superspreading and the effect of individual variation on
# disease emergence", Nature 438(7066), 355-359,
# doi:10.1038/nature04153. The negative binomial offspring
# parameterisation with dispersion k that this model uses.
#
# Native implementation mirroring Python morie.fn.ttrace exactly: the
# same gamma-Poisson mixture, the same Marsaglia-Tsang with the shape
# < 1 boost, the same Knuth/500-cutoff Poisson, the same serial
# interval draws, the same control definition (extinct within the
# horizon without exceeding the case cap), and the same
# effective_reproduction_number estimator that consumes uniforms in
# the same order the Python arm does.

.TT_EPS <- 1e-12

#' negbinom_offspring
#'
#' A step of the ttrace_native implementation. Called by \code{simulate_outbreak}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param R0 Coerced to numeric by the body, with \code{as.numeric}.
#' @param dispersion Coerced to numeric by the body, with \code{as.numeric}.
#' @param rng Passed to \code{.tt_gamma_draw}.
#' @return The value of \code{.tt_poisson_draw}.
#' @export
negbinom_offspring <- function(R0, dispersion, rng) {
  r0 <- as.numeric(R0)
  kk <- as.numeric(dispersion)
  if (r0 < 0)
    stop(sprintf("ttrace: R0 must be non-negative, got %r", R0))
  if (kk <= 0)
    stop(sprintf("ttrace: the dispersion k must be positive, got %r",
                 dispersion))
  if (r0 <= .TT_EPS)
    return(0L)
  if (kk > 1e6) {
    lam <- r0
  } else {
    lam <- .tt_gamma_draw(kk, r0 / kk, rng)
  }
  .tt_poisson_draw(lam, rng)
}

#' .tt_gamma_draw
#'
#' A step of the ttrace_native implementation. Called by \code{negbinom_offspring}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param shape Coerced to numeric by the body, with \code{as.numeric}.
#' @param scale Passed to \code{.tt_gamma_draw}.
#' @param rng Passed to \code{.ghc_unif}.
#' @return The value of \code{repeat}.
#' @export
.tt_gamma_draw <- function(shape, scale, rng) {
  a <- as.numeric(shape)
  if (a < 1) {
    u <- max(.ghc_unif(rng, 1L), 1e-300)
    return(.tt_gamma_draw(a + 1, scale, rng) * u^(1 / a))
  }
  d <- a - 1 / 3
  c_ <- 1 / sqrt(9 * d)
  repeat {
    x <- .ghc_norm(rng, 1L)[1L]
    v <- (1 + c_ * x)^3
    if (v <= 0)
      next
    u <- max(.ghc_unif(rng, 1L), 1e-300)
    if (log(u) < 0.5 * x * x + d - d * v + d * log(v))
      return(d * v * as.numeric(scale))
  }
}

#' .tt_poisson_draw
#'
#' A step of the ttrace_native implementation. Called by \code{negbinom_offspring}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param lam Coerced to numeric by the body, with \code{as.numeric}.
#' @param rng Passed to \code{.ghc_norm}.
#' @return The value of \code{repeat}.
#' @export
.tt_poisson_draw <- function(lam, rng) {
  lm <- as.numeric(lam)
  if (lm <= 0)
    return(0L)
  if (lm > 500) {
    z <- .ghc_norm(rng, 1L)[1L]
    return(max(0L, as.integer(round(lm + sqrt(lm) * z))))
  }
  L <- exp(-lm)
  n <- 0L
  p <- 1
  repeat {
    p <- p * max(.ghc_unif(rng, 1L), 1e-300)
    if (p <= L)
      return(n)
    n <- n + 1L
    if (n > 100000L)
      return(n)
  }
}

#' serial_interval_draw
#'
#' A step of the ttrace_native implementation. Called by \code{simulate_outbreak}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param mean Coerced to numeric by the body, with \code{as.numeric}.
#' @param sd Coerced to numeric by the body, with \code{as.numeric}.
#' @param rng Passed to \code{.ghc_norm}.
#' @param allow_presymptomatic A flag; the body branches on it. Defaults to \code{TRUE}.
#' @return The value of \code{v}, as built in the body.
#' @export
serial_interval_draw <- function(mean, sd, rng, allow_presymptomatic = TRUE) {
  m <- as.numeric(mean)
  s <- as.numeric(sd)
  if (s <= 0)
    stop("ttrace: the serial-interval sd must be positive")
  v <- m + s * .ghc_norm(rng, 1L)[1L]
  if (!isTRUE(allow_presymptomatic))
    return(max(v, 0))
  v
}

#' A "case" is a 3-tuple (infection_time, isolation_time, subclinical)
#'
#' A step of the ttrace_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param ... Passed through.
#' @return The value of \code{list}.
#' @export
.tt_active <- function(...) {
  # a "case" is a 3-tuple (infection_time, isolation_time, subclinical)
  list(...)
}

#' simulate_outbreak
#'
#' A step of the ttrace_native implementation. Called by \code{morie_ttrace}, \code{probability_of_control}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param R0 Passed to \code{negbinom_offspring}. Defaults to \code{2.5}.
#' @param dispersion Passed to \code{negbinom_offspring}. Defaults to \code{0.16}.
#' @param n_initial Coerced to integer by the body, with \code{as.integer}. Defaults to \code{20}.
#' @param trace_prob Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{0.8}.
#' @param delay_mean Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{3.83}.
#' @param delay_sd Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{2.4}.
#' @param si_mean Passed to \code{serial_interval_draw}. Defaults to \code{4.7}.
#' @param si_sd Passed to \code{serial_interval_draw}. Defaults to \code{2.9}.
#' @param subclinical Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{0}.
#' @param max_cases Coerced to integer by the body, with \code{as.integer}. Defaults to \code{5000}.
#' @param max_weeks Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{12}.
#' @param seed Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{0}.
#' @param allow_presymptomatic Passed to \code{serial_interval_draw}. Defaults to \code{TRUE}.
#' @return A list with \code{controlled}, \code{total_cases}, \code{weekly}, \code{hit_cap}, \code{extinct}.
#' @export
simulate_outbreak <- function(R0 = 2.5, dispersion = 0.16, n_initial = 20,
                              trace_prob = 0.8, delay_mean = 3.83,
                              delay_sd = 2.4, si_mean = 4.7, si_sd = 2.9,
                              subclinical = 0, max_cases = 5000,
                              max_weeks = 12, seed = 0,
                              allow_presymptomatic = TRUE) {
  rng <- .ghc_rng(as.numeric(seed))
  if (!(as.numeric(trace_prob) >= 0 && as.numeric(trace_prob) <= 1))
    stop(sprintf("ttrace: trace_prob must lie in [0, 1], got %r",
                 trace_prob))
  if (!(as.numeric(subclinical) >= 0 && as.numeric(subclinical) <= 1))
    stop(sprintf("ttrace: subclinical must lie in [0, 1], got %r",
                 subclinical))
  if (as.integer(n_initial) < 1L)
    stop("ttrace: need at least one initial case")
  horizon <- as.numeric(max_weeks) * 7

  active <- vector("list", as.integer(n_initial))
  for (i in seq_len(as.integer(n_initial))) {
    sub <- .ghc_unif(rng, 1L) < as.numeric(subclinical)
    iso <- if (sub) Inf
           else max(0, as.numeric(delay_mean)
                    + as.numeric(delay_sd) * .ghc_norm(rng, 1L)[1L])
    active[[i]] <- list(0, iso, sub)
  }
  total <- as.integer(n_initial)
  weekly <- rep(0L, as.integer(max_weeks) + 1L)
  weekly[1L] <- as.integer(n_initial)
  hit_cap <- FALSE

  while (length(active) > 0L) {
    nxt <- list()
    for (case in active) {
      t_inf <- case[[1L]]
      t_iso <- case[[2L]]
      n_off <- negbinom_offspring(R0, dispersion, rng)
      for (j in seq_len(n_off)) {
        si <- serial_interval_draw(si_mean, si_sd, rng,
                                   allow_presymptomatic = allow_presymptomatic)
        t_new <- t_inf + si
        if (t_new < t_inf)
          next
        if (t_new >= t_iso)
          next
        if (t_new > horizon)
          next
        sub <- .ghc_unif(rng, 1L) < as.numeric(subclinical)
        traced <- (!sub) && (.ghc_unif(rng, 1L) < as.numeric(trace_prob))
        if (sub) {
          iso_new <- Inf
        } else if (traced) {
          iso_new <- max(t_new, t_iso)
        } else {
          iso_new <- t_new + max(0, as.numeric(delay_mean)
                                  + as.numeric(delay_sd)
                                    * .ghc_norm(rng, 1L)[1L])
        }
        nxt[[length(nxt) + 1L]] <- list(t_new, iso_new, sub)
        total <- total + 1L
        wk <- as.integer(floor(t_new / 7))
        if (wk >= 0L && wk <= as.integer(max_weeks))
          weekly[wk + 1L] <- weekly[wk + 1L] + 1L
        if (total > as.integer(max_cases)) {
          hit_cap <- TRUE
          break
        }
      }
      if (hit_cap)
        break
    }
    if (hit_cap)
      break
    active <- nxt
  }

  controlled <- (!hit_cap) && (length(active) == 0L)
  list(controlled = controlled, total_cases = total, weekly = weekly,
       hit_cap = hit_cap, extinct = length(active) == 0L)
}

#' probability_of_control
#'
#' A step of the ttrace_native implementation. Called by \code{morie_ttrace}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param reps Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{200}.
#' @param seed Coerced to integer by the body, with \code{as.integer}. Defaults to \code{0}.
#' @param ... Passed through.
#' @return A list with \code{estimate}, \code{probability_of_control}, \code{se}, \code{reps}, \code{median_size}, \code{max_size}, \code{max_cases}, \code{max_weeks}, \code{definition}, \code{method}.
#' @export
probability_of_control <- function(reps = 200, seed = 0, ...) {
  ok <- 0L
  sizes <- integer(as.integer(reps))
  for (r in seq_len(as.integer(reps)) - 1L) {
    out <- simulate_outbreak(seed = as.integer(seed) * 7919L + r, ...)
    if (out$controlled)
      ok <- ok + 1L
    sizes[r + 1L] <- out$total_cases
  }
  p <- ok / as.numeric(reps)
  se <- sqrt(max(p * (1 - p), 0) / as.numeric(reps))
  sizes <- sort(sizes)
  list(estimate = p, probability_of_control = p, se = se,
       reps = as.integer(reps),
       median_size = sizes[length(sizes) %/% 2L + 1L],
       max_size = sizes[length(sizes)],
       max_cases = if (!is.null(list(...)$max_cases)) list(...)$max_cases
                   else 5000,
       max_weeks = if (!is.null(list(...)$max_weeks)) list(...)$max_weeks
                   else 12,
       definition = paste0("extinct within max_weeks without exceeding ",
                           "max_cases; both change the answer"),
       method = paste("branching-process simulation, Hellewell et al. ",
                      "(2020) Methods"))
}

#' effective_reproduction_number
#'
#' A step of the ttrace_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param R0 Coerced to numeric by the body, with \code{as.numeric}.
#' @param si_mean Coerced to numeric by the body, with \code{as.numeric}.
#' @param si_sd Coerced to numeric by the body, with \code{as.numeric}.
#' @param delay_mean Coerced to numeric by the body, with \code{as.numeric}.
#' @param delay_sd Coerced to numeric by the body, with \code{as.numeric}.
#' @param trace_prob Coerced to numeric by the body, with \code{as.numeric}.
#' @param subclinical Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{0}.
#' @param draws Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{20000}.
#' @param seed Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{0}.
#' @return A list with \code{R_eff}, \code{R0}, \code{fraction_before_isolation}, \code{controlled_in_expectation}, \code{note}.
#' @export
effective_reproduction_number <- function(R0, si_mean, si_sd, delay_mean,
                                          delay_sd, trace_prob,
                                          subclinical = 0, draws = 20000,
                                          seed = 0) {
  rng <- .ghc_rng(as.numeric(seed))
  hit <- 0L
  for (i in seq_len(as.integer(draws))) {
    if (.ghc_unif(rng, 1L) < as.numeric(subclinical)) {
      hit <- hit + 1L
      next
    }
    traced <- .ghc_unif(rng, 1L) < as.numeric(trace_prob)
    t_iso <- if (traced) 0
             else max(0, as.numeric(delay_mean)
                      + as.numeric(delay_sd) * .ghc_norm(rng, 1L)[1L])
    si <- as.numeric(si_mean) + as.numeric(si_sd) * .ghc_norm(rng, 1L)[1L]
    if (si < t_iso)
      hit <- hit + 1L
  }
  frac <- hit / as.numeric(draws)
  list(R_eff = as.numeric(R0) * frac, R0 = as.numeric(R0),
       fraction_before_isolation = frac,
       controlled_in_expectation = as.numeric(R0) * frac < 1,
       note = paste("a traced contact is quarantined when its infector ",
                    "is isolated, so its own transmission window is ",
                    "measured from that point"))
}

#' .ttrace_cheatsheet
#'
#' A step of the ttrace_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @return A character value.
#' @export
.ttrace_cheatsheet <- function() {
  paste0("ttrace: branching process. Offspring ~ NegBinom(mean R0, ",
         "dispersion k), variance R0(1 + R0/k) -- overdispersion ",
         "matters because small k means most chains die alone. A ",
         "secondary case exists ONLY if the infector was not yet ",
         "isolated. So the lever is the fraction of the serial ",
         "interval falling before isolation, which is why ",
         "PRESYMPTOMATIC transmission decides feasibility. ",
         "Subclinical cases are never isolated at all -- a hard ",
         "ceiling no amount of tracing clears.")
}

# compact alias per ledger/NAMING.md
contacttracingyield <- probability_of_control

# public names resolved by the lazy map
contact_tracing_yield <- probability_of_control

#' morie_ttrace
#'
#' A step of the ttrace_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param seed Passed to \code{simulate_outbreak}. Defaults to \code{0}.
#' @param R0 Passed to \code{simulate_outbreak}. Defaults to \code{2.5}.
#' @param dispersion Passed to \code{simulate_outbreak}. Defaults to \code{0.16}.
#' @param n_initial Passed to \code{simulate_outbreak}. Defaults to \code{20}.
#' @param trace_prob Passed to \code{simulate_outbreak}. Defaults to \code{0.8}.
#' @param delay_mean Passed to \code{simulate_outbreak}. Defaults to \code{3.83}.
#' @param delay_sd Passed to \code{simulate_outbreak}. Defaults to \code{2.4}.
#' @param si_mean Passed to \code{simulate_outbreak}. Defaults to \code{4.7}.
#' @param si_sd Passed to \code{simulate_outbreak}. Defaults to \code{2.9}.
#' @param subclinical Passed to \code{simulate_outbreak}. Defaults to \code{0}.
#' @param max_cases Passed to \code{simulate_outbreak}. Defaults to \code{5000}.
#' @param max_weeks Passed to \code{simulate_outbreak}. Defaults to \code{12}.
#' @param allow_presymptomatic Passed to \code{simulate_outbreak}. Defaults to \code{TRUE}.
#' @param reps Passed to \code{probability_of_control}. Defaults to \code{200}.
#' @return A list with \code{simulation}, \code{probability_of_control}.
#' @export
morie_ttrace <- function(seed = 0, R0 = 2.5, dispersion = 0.16,
                         n_initial = 20, trace_prob = 0.8,
                         delay_mean = 3.83, delay_sd = 2.4, si_mean = 4.7,
                         si_sd = 2.9, subclinical = 0, max_cases = 5000,
                         max_weeks = 12, allow_presymptomatic = TRUE,
                         reps = 200) {
  single <- simulate_outbreak(R0 = R0, dispersion = dispersion,
                              n_initial = n_initial, trace_prob = trace_prob,
                              delay_mean = delay_mean, delay_sd = delay_sd,
                              si_mean = si_mean, si_sd = si_sd,
                              subclinical = subclinical, max_cases = max_cases,
                              max_weeks = max_weeks, seed = seed,
                              allow_presymptomatic = allow_presymptomatic)
  pc <- probability_of_control(R0 = R0, dispersion = dispersion,
                               n_initial = n_initial, trace_prob = trace_prob,
                               delay_mean = delay_mean, delay_sd = delay_sd,
                               si_mean = si_mean, si_sd = si_sd,
                               subclinical = subclinical, max_cases = max_cases,
                               max_weeks = max_weeks,
                               allow_presymptomatic = allow_presymptomatic,
                               reps = reps, seed = seed)
  list(simulation = single, probability_of_control = pc)
}
