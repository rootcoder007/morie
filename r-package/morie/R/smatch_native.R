# SCCS design matrix, Poisson fit, sample size, power and efficiency.
# Sources: Whitaker, H. J., Farrington, C. P., Spiessens, B. & Musonda,
# P. (2006) "Tutorial in biostatistics: The self-controlled case
# series method", Statistics in Medicine 25, 1768-1797,
# doi:10.1002/sim.2302; Farrington, C. P. (1995) "Relative Incidence
# Estimation from Case Series for Vaccine Safety Evaluation",
# Biometrics 51(1), 228-235; Musonda, P., Farrington, C. P. &
# Whitaker, H. J. (2006) "Sample sizes for self-controlled case series
# studies", Statistics in Medicine 25(15), 2618-2631.
#
# Native implementation mirroring morie.fn.smatch exactly: the same
# Sec. 4 design (n, log offset, per-individual factor columns), the
# same IRLS Poisson fit and the same Sec. 7.6 sample-size expression.

.SMA_EPS <- 1e-12

#' Build the SCCS Poisson design matrix (Sec. 4)
#'
#' Counts, log-time offset, and factor columns for risk period, age
#' band, and one per individual. The per-individual factors force the
#' fitted totals to equal the observed ones, which is the conditioning
#' the multinomial fit performs.
#'
#' @param cases List of case dicts with \code{start, end, events,
#'   exposure}.
#' @param risk_periods List of \code{c(start, end)} in age units.
#' @param age_breaks Numeric vector of age-band boundaries.
#' @return List with \code{y, offset, X, n_risk, n_age, n_people, n_rows}.
#' @export
morie_smatch_poisson_design <- function(cases, risk_periods,
                                         age_breaks = numeric(0)) {
  rp <- lapply(risk_periods, function(r) c(as.numeric(r[1]), as.numeric(r[2])))
  ab <- as.numeric(age_breaks)
  n_risk <- length(rp); n_age <- length(ab) + 1L
  # Build intervals for each case
  people <- list()
  for (c in cases) {
    ev <- c$events %||% list()
    if (length(ev) == 0L) next
    cells <- .smatch_build_intervals(c$start, c$end, c$exposure %||% NULL,
                                      ev, rp, ab)
    people[[length(people) + 1L]] <- cells
  }
  if (length(people) == 0L) stop("smatch: no case contributed an event")
  P <- length(people)
  ncol <- n_risk + (n_age - 1L) + P
  y <- numeric(0); off <- numeric(0); X <- matrix(0, 0, ncol)
  for (i in seq_along(people)) {
    cells <- people[[i]]
    for (cell in cells) {
      j <- cell$age; r <- cell$risk; e <- cell$exposure; n <- cell$n
      if (e <= .SMA_EPS) next
      row <- numeric(ncol)
      if (r > 0L) row[r] <- 1
      if (j > 0L) row[n_risk + j] <- 1
      row[n_risk + n_age - 1L + i] <- 1
      X <- rbind(X, row)
      y <- c(y, n); off <- c(off, log(e))
    }
  }
  list(y = y, offset = off, X = X, n_risk = n_risk, n_age = n_age,
       n_people = P, n_rows = length(y))
}

#' Build (age, risk, exposure_time, n) cells for one case
#' @keywords internal
#' @noRd
.smatch_build_intervals <- function(start, end, exposure, events, rp,
                                      ab) {
  total_dur <- end - start
  boundaries <- c(0, ab, total_dur)
  # risk windows relative to case start
  risk_int <- lapply(rp, function(r) c(max(0, r[1]), min(total_dur, r[2])))
  out <- list()
  for (b in seq_len(length(boundaries) - 1L)) {
    lo <- boundaries[b]; hi <- boundaries[b + 1L]
    if (hi <= lo) next
    age <- b - 1L
    in_risk <- which(sapply(risk_int, function(ri) !(ri[2] <= lo || ri[1] >= hi)))
    if (length(in_risk) == 0L) {
      dur <- hi - lo
      ne <- sum(sapply(events, function(t) t > lo && t <= hi))
      out[[length(out) + 1L]] <- list(age = age, risk = 0L,
                                       exposure = dur, n = ne)
    } else {
      # split into exposed and unexposed portions within this age band
      cuts <- sort(unique(c(lo, hi,
                            unlist(lapply(risk_int[in_risk],
                                          function(ri) c(ri[1], ri[2]))))))
      cuts <- cuts[cuts > lo & cuts < hi]
      for (k in seq_len(length(cuts) - 1L)) {
        a <- cuts[k]; z <- cuts[k + 1L]
        mid <- (a + z) / 2
        is_risk <- any(sapply(risk_int[in_risk], function(ri) ri[1] <= mid && ri[2] >= mid))
        dur <- z - a
        ne <- sum(sapply(events, function(t) t > a && t <= z))
        out[[length(out) + 1L]] <- list(age = age, risk = if (is_risk) 1L else 0L,
                                         exposure = dur, n = ne)
      }
    }
  }
  out
}

#' IRLS Poisson fit of the SCCS design
#'
#' Returns the same relative incidences the conditional multinomial
#' fit returns: the per-individual factors are nuisance parameters
#' whose only job is to reproduce the observed totals.
#'
#' @param cases List of case dicts.
#' @param risk_periods List of \code{c(start, end)} windows.
#' @param age_breaks Numeric vector.
#' @param iters Maximum IRLS iterations.
#' @tol Convergence tolerance on max parameter change.
#' @param ridge Ridge added to the normal equation.
#' @return List with \code{relative_incidence} (==\code{estimate}),
#'   \code{log_ri}, \code{age_effects}, \code{individual_effects},
#'   \code{coef}, \code{converged}, \code{iterations}, \code{n_rows},
#'   \code{n_people}, \code{method}, \code{identical_to}.
#' @export
morie_smatch_sccs_poisson_fit <- function(cases, risk_periods,
                                           age_breaks = numeric(0),
                                           iters = 200, tol = 1e-12,
                                           ridge = 1e-9) {
  d <- morie_smatch_poisson_design(cases, risk_periods, age_breaks)
  y <- d$y; off <- d$offset; X <- d$X; p <- ncol(X)
  beta <- rep(0, p); conv <- FALSE; it <- 0L
  for (kk in seq_len(as.integer(iters))) {
    it <- kk
    eta <- off + as.numeric(X %*% beta)
    eta <- pmin(pmax(eta, -500), 500)
    mu <- exp(eta)
    W <- pmax(mu, 1e-12)
    z <- eta - off + (y - mu) / W
    XtWX <- crossprod(X, X * W)
    XtWz <- as.numeric(crossprod(X, W * z))
    diag(XtWX) <- diag(XtWX) + ridge
    nb <- tryCatch(solve(XtWX, XtWz), error = function(e)
      stop("smatch: the Poisson design is singular -- an interval has no exposure time or an individual has no variation"))
    if (max(abs(nb - beta)) < tol) { beta <- nb; conv <- TRUE; break }
    beta <- nb
  }
  nr <- d$n_risk
  list(estimate = exp(beta[seq_len(nr)]),
       relative_incidence = exp(beta[seq_len(nr)]),
       log_ri = beta[seq_len(nr)],
       age_effects = beta[seq.int(nr + 1L, nr + d$n_age - 1L)],
       individual_effects = beta[seq.int(nr + d$n_age,
                                          nr + d$n_age - 1L + d$n_people)],
       coef = beta, converged = conv, iterations = it,
       n_rows = d$n_rows, n_people = d$n_people,
       method = "associated Poisson model with a per-individual factor and log-time offset; Whitaker et al. (2006) Sec. 4",
       identical_to = "the conditional multinomial fit of sccsno")
}

#' Events required to detect a log relative incidence (Sec. 7.6)
#'
#' Assumes age effects are negligible. \code{r} is the ratio of risk
#' period to observation period; \code{p_exposed} is the proportion of
#' the POPULATION exposed.
#'
#' @param log_ri Effect size on the log scale.
#' @param r Risk-period fraction.
#' @param p_exposed Population exposed fraction.
#' @param alpha Two-sided alpha.
#' @param power Target power.
#' @return List with \code{n_events}, \code{n_events_ceiling},
#'   \code{rho, A, B, C, z_alpha_2, z_power, ...}.
#' @export
morie_smatch_sample_size <- function(log_ri, r, p_exposed,
                                      alpha = 0.05, power = 0.8) {
  b <- as.numeric(log_ri); rr <- as.numeric(r); p <- as.numeric(p_exposed)
  if (b == 0) stop("smatch: the sample size is unbounded at a log relative incidence of 0")
  if (rr <= 0 || rr >= 1)
    stop("smatch: r must lie strictly in (0, 1), got ", rr,
         " -- it is the risk period as a fraction of the observation period")
  if (p <= 0 || p > 1)
    stop("smatch: p_exposed must lie in (0, 1], got ", p)
  if (alpha <= 0 || alpha >= 1) stop("smatch: alpha must lie in (0, 1)")
  if (power <= 0 || power >= 1) stop("smatch: power must lie in (0, 1)")
  eb <- exp(b); den <- rr * eb + 1 - rr
  rho <- rr * eb / den
  A <- 2 * (rho * b - log(den))
  if (A <= .SMA_EPS)
    stop(sprintf("smatch: the information A is non-positive (%.3e) -- the design carries no signal here", A))
  B <- b * b * rho * (1 - rho) / A
  C <- 1 + (1 - p) / (p * den)
  za <- .smatch_qnorm(1 - alpha / 2)
  zg <- .smatch_qnorm(power)
  n <- (C / A) * (za + zg * sqrt(B))^2
  list(n_events = n, n_events_ceiling = as.integer(ceiling(n)),
       rho = rho, A = A, B = B, C = C, z_alpha_2 = za, z_power = zg,
       log_ri = b, r = rr, p_exposed = p,
       assumes = "age effects negligible; see Musonda, Farrington & Whitaker (2006) otherwise",
       method = "Whitaker et al. (2006) Sec. 7.6")
}

#' Power at a given number of events
#' @export
morie_smatch_power <- function(n_events, log_ri, r, p_exposed,
                                alpha = 0.05) {
  s <- morie_smatch_sample_size(log_ri, r, p_exposed, alpha = alpha,
                                 power = 0.5)
  A <- s$A; B <- s$B; C <- s$C; za <- s$z_alpha_2
  root <- sqrt(max(n_events * A / C, 0))
  zg <- if (B > .SMA_EPS) (root - za) / sqrt(B) else Inf
  list(power = pnorm(zg), z_power = zg, n_events = n_events,
       A = A, B = B, C = C)
}

#' Asymptotic efficiency against the cohort design (Sec. 7.5)
#' @export
morie_smatch_relative_efficiency <- function(r, log_ri) {
  rr <- as.numeric(r); b <- as.numeric(log_ri)
  if (rr <= 0 || rr >= 1) stop("smatch: r must lie strictly in (0, 1)")
  eb <- exp(b); den <- rr * eb + 1 - rr
  rho <- rr * eb / den
  list(rho = rho, efficiency = 1 - rho, r = rr, log_ri = b,
       interpretation = "the fraction of cases falling in the risk period is rho; the marginal information lost grows with it, so a SHORT risk period keeps efficiency high (Sec. 7.5)")
}

# Beasley-Springer-Moro inverse normal CDF
.smatch_qnorm <- function(p) {
  p <- pmin(pmax(p, 1e-15), 1 - 1e-15)
  a <- c(-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
  b <- c(-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
  c <- c(-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
  d <- c(7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
  plow <- p < 0.02425
  phigh <- p > 1 - 0.02425
  out <- numeric(length(p))
  if (any(plow)) {
    q <- sqrt(-2 * log(p[plow]))
    out[plow] <- (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) *
                    q + c[6]) /
      ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
  }
  if (any(phigh)) {
    q <- sqrt(-2 * log(1 - p[phigh]))
    out[phigh] <- -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) *
                      q + c[6]) /
      ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
  }
  if (any(!plow & !phigh)) {
    q <- p[!plow & !phigh] - 0.5
    r <- q * q
    out[!plow & !phigh] <- (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) *
                               r + a[5]) * r + a[6]) * q /
      (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1)
  }
  out
}

`%||%` <- function(x, y) if (is.null(x)) y else x

# house entry point: the package exports one morie_<module>
morie_smatch <- morie_smatch_poisson_design
