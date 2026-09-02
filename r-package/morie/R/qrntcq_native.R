# Quarantine efficacy: how much transmission a duration actually
# prevents.
# Sources: Ashcroft, P., Lehtinen, S., Angst, D. C., Low, N. &
# Bonhoeffer, S. (2021) "Quantifying the impact of quarantine duration
# on COVID-19 transmission", eLife 10, e63704, doi:10.7554/eLife.63704
# (eq. 1: efficacy as the fraction of remaining transmission; eq. 2:
# test-and-release efficacy; eq. 4: utility as efficacy per day);
# Kucirka, L. M., Lauer, S. A., Laeyendecker, O., Boon, D. & Lessler,
# J. (2020) "Variation in false-negative rate of reverse transcriptase
# polymerase chain reaction-based SARS-CoV-2 tests by time since
# exposure", Annals of Internal Medicine 173(4), 262-267.
#
# Native implementation mirroring morie.fn.qrntcq exactly: the same
# gamma generation-time density on a grid, the same efficacy as
# mass of the density between t_Q and t_R over the mass remaining
# after t_Q (so there is a ceiling below 1 whenever quarantine starts
# late), the same test-and-release efficacy averaged over the
# false-negative and positive branches, and the same utility whose
# ratio is independent of the infected fraction for standard
# quarantine.

.qrntcq_EPS <- 1e-12

#' @keywords internal
#' @noRd
.lgamma <- function(x) lgamma(x)

#' Build a gamma generation-time density on a grid, normalised
#'
#' Defaults are a shape/scale pair in the range reported for
#' SARS-CoV-2; they are a placeholder, not the paper's fit.
#'
#' @param shape Positive shape.
#' @param scale Positive scale.
#' @param grid Optional grid of times.
#' @param t_max Maximum time when \code{grid} is \code{NULL}.
#' @param n Number of grid points.
#' @return A list with \code{t} and \code{density}.
#' @export
gamma_generation_time <- function(shape = 2.83, scale = 1.86, grid = NULL,
                                  t_max = 30.0, n = 3001L) {
  if (as.numeric(shape) <= 0 || as.numeric(scale) <= 0)
    stop("qrntcq: the gamma shape and scale must be positive")
  if (is.null(grid)) {
    ts <- seq(0.0, as.numeric(t_max), length.out = as.integer(n))
  } else {
    ts <- as.numeric(grid)
  }
  a <- as.numeric(shape); b <- as.numeric(scale)
  dens <- numeric(length(ts))
  for (i in seq_along(ts)) {
    t <- ts[i]
    if (t <= 0) {
      dens[i] <- 0.0
    } else {
      dens[i] <- exp((a - 1.0) * log(t) - t / b - .lgamma(a) - a * log(b))
    }
  }
  z <- .trapz(ts, dens)
  if (z <= .qrntcq_EPS)
    stop("qrntcq: the generation-time density integrates to zero")
  list(t = ts, density = as.numeric(dens) / z)
}

#' @param ts See Usage.
#' @param ys See Usage.
#' @keywords internal
.trapz <- function(ts, ys) {
  tot <- 0.0
  for (i in seq_len(length(ts) - 1L))
    tot <- tot + 0.5 * (ys[i] + ys[i + 1L]) * (ts[i + 1L] - ts[i])
  tot
}

#' @param ts See Usage.
#' @param ys See Usage.
#' @param lo See Usage.
#' @param hi See Usage.
#' @keywords internal
.mass <- function(ts, ys, lo, hi) {
  if (hi <= lo) return(0.0)
  tot <- 0.0
  for (i in seq_len(length(ts) - 1L)) {
    a <- ts[i]; b <- ts[i + 1L]
    if (b <= lo || a >= hi) next
    l <- max(a, lo); r <- min(b, hi)
    if (r <= l) next
    w <- b - a
    if (w > 0) {
      ya <- ys[i] + (ys[i + 1L] - ys[i]) * ((l - a) / w)
      yb <- ys[i] + (ys[i + 1L] - ys[i]) * ((r - a) / w)
      tot <- tot + 0.5 * (ya + yb) * (r - l)
    }
  }
  tot
}

#' Quarantine efficacy (eq. 1 of Ashcroft et al. 2021)
#'
#' \code{efficacy = mass of the density in \[t_Q, t_R\] /
#' mass remaining after t_Q}. Transmission before \code{t_Q} is
#' already gone, so the denominator is the mass from \code{t_Q} on.
#'
#' @param t_Q Quarantine start.
#' @param t_R Release time.
#' @param generation_time Output of \code{\link{gamma_generation_time}}.
#' @param t_E Exposure time.
#' @return A list with \code{efficacy}, \code{prevented_mass},
#'   \code{remaining_mass}, \code{t_Q}, \code{t_R}, \code{max_attainable},
#'   \code{pre_quarantine_mass} (and a \code{note} if no transmission
#'   remains).
#' @export
quarantine_efficacy <- function(t_Q, t_R, generation_time = NULL,
                                t_E = 0.0) {
  g <- if (is.null(generation_time)) gamma_generation_time()
       else generation_time
  ts <- g$t; ys <- g$density
  q <- as.numeric(t_Q); r <- as.numeric(t_R)
  if (r < q)
    stop(sprintf("qrntcq: release at %g precedes quarantine ",
                 "start at %g", r, q))
  if (q < as.numeric(t_E))
    stop(sprintf("qrntcq: quarantine cannot start before ",
                 "exposure (t_Q %g < t_E %g)", q, as.numeric(t_E)))
  remaining <- .mass(ts, ys, q, ts[length(ts)])
  if (remaining <= .qrntcq_EPS) {
    return(list(efficacy = 0.0, remaining_mass = remaining,
                prevented_mass = 0.0,
                note = paste0("no transmission remains after t_Q, so ",
                              "quarantine can prevent nothing")))
  }
  prevented <- .mass(ts, ys, q, r)
  list(efficacy = prevented / remaining,
       prevented_mass = prevented, remaining_mass = remaining,
       t_Q = q, t_R = r, max_attainable = 1.0,
       pre_quarantine_mass = .mass(ts, ys, ts[1L], q))
}

#' Test-and-release efficacy (eq. 2 of Ashcroft et al. 2021)
#'
#' Averaged over those who test negative and are released at
#' \code{t_R} and those who test positive and are detained until
#' \code{t_R_positive}.
#'
#' @param t_Q Quarantine start.
#' @param t_T Test time.
#' @param t_R Release time.
#' @param false_negative False-negative probability in \code{\[0, 1\]}.
#' @param generation_time Output of \code{\link{gamma_generation_time}}.
#' @param t_R_positive Release time for test-positives; defaults to
#'   the end of the grid.
#' @return A list with \code{efficacy}, \code{efficacy_detained},
#'   \code{efficacy_released}, \code{false_negative}, \code{t_T},
#'   \code{t_R}, \code{bound}, \code{note}.
#' @export
efficacy_test_and_release <- function(t_Q, t_T, t_R, false_negative,
                                      generation_time = NULL,
                                      t_R_positive = NULL) {
  g <- if (is.null(generation_time)) gamma_generation_time()
       else generation_time
  p <- as.numeric(false_negative)
  if (p < 0 || p > 1)
    stop(sprintf("qrntcq: the false-negative probability must ",
                 "lie in [0, 1], got %r", false_negative))
  if (as.numeric(t_T) < as.numeric(t_Q))
    stop("qrntcq: the test cannot precede the start of quarantine")
  if (as.numeric(t_R) < as.numeric(t_T))
    stop("qrntcq: release cannot precede the test")
  stay <- if (is.null(t_R_positive)) g$t[length(g$t)]
          else as.numeric(t_R_positive)
  released <- quarantine_efficacy(t_Q, t_R, g)$efficacy
  detained <- quarantine_efficacy(t_Q, stay, g)$efficacy
  eff <- (1.0 - p) * detained + p * released
  list(efficacy = eff, efficacy_detained = detained,
       efficacy_released = released, false_negative = p,
       t_T = as.numeric(t_T), t_R = as.numeric(t_R),
       bound = detained,
       note = paste0("always at or below the efficacy of detaining ",
                     "everyone until t_R_positive, because a false ",
                     "negative releases an infectious person"))
}

#' Utility = efficacy per person-day in quarantine (eq. 4)
#' @param efficacy Numeric efficacy.
#' @param days_in_quarantine Positive number of days.
#' @return Numeric.
#' @export
utility <- function(efficacy, days_in_quarantine) {
  d <- as.numeric(days_in_quarantine)
  if (d <= 0)
    stop("qrntcq: the time in quarantine must be positive")
  as.numeric(efficacy) / d
}

#' Utility of one standard quarantine relative to another
#'
#' The infected fraction cancels in the ratio for standard
#' quarantine; \code{infected_fraction} is accepted only to
#' demonstrate the cancellation.
#'
#' @param t_R_a Release time of strategy A.
#' @param t_R_b Release time of strategy B.
#' @param t_Q Quarantine start.
#' @param generation_time Output of \code{\link{gamma_generation_time}}.
#' @param infected_fraction Optional, ignored in the computation.
#' @return A list with \code{relative_utility}, \code{utility_a},
#'   \code{utility_b}, \code{efficacy_a}, \code{efficacy_b},
#'   \code{independent_of_infected_fraction}, \code{note}.
#' @export
relative_utility <- function(t_R_a, t_R_b, t_Q = 3.0,
                             generation_time = NULL,
                             infected_fraction = NULL) {
  g <- if (is.null(generation_time)) gamma_generation_time()
       else generation_time
  ea <- quarantine_efficacy(t_Q, t_R_a, g)$efficacy
  eb <- quarantine_efficacy(t_Q, t_R_b, g)$efficacy
  da <- as.numeric(t_R_a) - as.numeric(t_Q)
  db <- as.numeric(t_R_b) - as.numeric(t_Q)
  if (da <= 0 || db <= 0)
    stop("qrntcq: both quarantines must have positive duration")
  list(relative_utility = (ea / da) / (eb / db),
       utility_a = ea / da, utility_b = eb / db,
       efficacy_a = ea, efficacy_b = eb,
       independent_of_infected_fraction = TRUE,
       note = paste0("the infected fraction cancels for standard ",
                     "quarantine, so 'most quarantined people are not ",
                     "infected' is not an argument for shortening it"))
}

#' Release time maximising utility
#'
#' @param t_Q Quarantine start.
#' @param generation_time Output of \code{\link{gamma_generation_time}}.
#' @param t_max Maximum release time to consider.
#' @param step Step size.
#' @return A list with \code{estimate}, \code{optimal_t_R},
#'   \code{efficacy_at_optimum}, \code{utility_at_optimum}, \code{curve},
#'   \code{t_Q}, \code{method}.
#' @export
optimal_duration <- function(t_Q = 3.0, generation_time = NULL,
                              t_max = 20.0, step = 0.25) {
  g <- if (is.null(generation_time)) gamma_generation_time()
       else generation_time
  best <- NULL
  curve <- list()
  t <- as.numeric(t_Q) + as.numeric(step)
  while (t <= as.numeric(t_max) + .qrntcq_EPS) {
    e <- quarantine_efficacy(t_Q, t, g)$efficacy
    u <- e / (t - as.numeric(t_Q))
    curve[[length(curve) + 1L]] <- list(t_R = t, efficacy = e,
                                         utility = u)
    if (is.null(best) || u > best$utility) {
      best <- list(t_R = t, efficacy = e, utility = u)
    }
    t <- t + as.numeric(step)
  }
  list(estimate = best$t_R, optimal_t_R = best$t_R,
       efficacy_at_optimum = best$efficacy,
       utility_at_optimum = best$utility,
       curve = curve, t_Q = as.numeric(t_Q),
       method = paste0("utility maximisation, ",
                       "Ashcroft et al. (2021) eq. (4)"))
}

#' @export
quarantineefficacy <- quarantine_efficacy

#' @export
testandrelease <- efficacy_test_and_release

#' @export
morie_qrntcq <- function(...) NULL
attr(morie_qrntcq, "info") <- paste0(
  "qrntcq: efficacy = mass of the generation-time density ",
  "between t_Q and t_R, over the mass remaining after t_Q. ",
  "Transmission before quarantine is unrecoverable, so ",
  "there is a CEILING every strategy sits under. ",
  "Test-and-release is always below it (false negatives ",
  "release infectious people) but wins on utility = ",
  "efficacy per day. For STANDARD quarantine the infected ",
  "fraction cancels in a utility ratio -- so 'most ",
  "quarantined people are not infected' is not an argument ",
  "for shortening.")

#' @export
qrntcq_cheatsheet <- function() {
  paste0("qrntcq: efficacy = mass of the generation-time density ",
         "between t_Q and t_R, over the mass remaining after t_Q. ",
         "Transmission before quarantine is unrecoverable, so ",
         "there is a CEILING every strategy sits under. ",
         "Test-and-release is always below it (false negatives ",
         "release infectious people) but wins on utility = ",
         "efficacy per day. For STANDARD quarantine the infected ",
         "fraction cancels in a utility ratio -- so 'most ",
         "quarantined people are not infected' is not an argument ",
         "for shortening.")
}
