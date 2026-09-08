# Prophet: piecewise trend, Fourier seasonality, holidays.
# Sources: Taylor, S. J. & Letham, B. (2018) "Forecasting at Scale",
# The American Statistician 72(1), 37-45, doi:10.1080/00031305.2017.1380080;
# preprint PeerJ Preprints 5:e3190v2, doi:10.7287/peerj.preprints.3190v2.
# Eq. (1)-(4), Secs. 3.1-3.3.
#
# Native implementation mirroring morie.fn.prphet exactly: the same
# piecewise-linear trend with gamma_j = -s_j delta_j so the segments
# join by construction, the same Fourier seasonality, the same per-holiday
# indicators widened by a window, and the same cyclic coordinate-descent
# fit with soft-thresholding on the deltas (the L1 sparsity IS the
# changepoint selection -- ridge cannot replicate it).

#' Place candidate changepoints
#'
#' If \code{changepoints} is supplied it is used as-is; otherwise
#' \code{n_changepoints} equally spaced points are placed over the
#' first \code{changepoint_range} of the time range.
#'
#' @param t Numeric vector of times.
#' @param n_changepoints Integer, number of candidate changepoints.
#' @param changepoint_range Fraction of the time range to span.
#' @param changepoints Optional explicit vector of changepoints.
#' @return Numeric vector of changepoints.
#' @export
.changepoints <- function(t, n_changepoints, changepoint_range = 0.8,
                          changepoints = NULL) {
  if (!is.null(changepoints)) return(as.numeric(changepoints))
  n <- length(t)
  hi <- t[1L] + changepoint_range * (t[length(t)] - t[1L])
  m <- as.integer(n_changepoints)
  if (m < 1L) return(numeric(0))
  step <- (hi - t[1L]) / (m + 1L)
  t[1L] + step * (seq_len(m))
}

#' Evaluate the piecewise-linear trend (Eq. 4 of Taylor & Letham 2018)
#'
#' The offsets carry \code{gamma_j = -s_j delta_j} so the segments join
#' by construction.
#'
#' @param t Numeric vector of times.
#' @param k_rate Initial rate.
#' @param m_off Initial offset.
#' @param deltas Numeric vector of rate adjustments, one per changepoint.
#' @param cps Numeric vector of changepoints.
#' @return Numeric vector of trend values.
#' @export
piecewise_trend <- function(t, k_rate, m_off, deltas, cps) {
  out <- numeric(length(t))
  for (i in seq_along(t)) {
    tv <- t[i]
    a <- ifelse(tv >= cps, 1.0, 0.0)
    rate <- k_rate + sum(a * deltas)
    off <- m_off + sum(a * (-cps * deltas))
    out[i] <- rate * tv + off
  }
  out
}

#' Design columns for k, m and each delta_j
#'
#' The delta column is \code{a_j(t)(t - s_j)}, which already carries
#' the \code{-s_j delta_j} offset -- so continuity holds by construction.
#'
#' @param t Numeric vector of times.
#' @param cps Numeric vector of changepoints.
#' @return A list of rows, each a numeric vector.
#' @export
trend_matrix <- function(t, cps) {
  rows <- list()
  for (tv in t) {
    row <- c(tv, 1.0)
    for (s in cps) row <- c(row, if (tv >= s) tv - s else 0.0)
    rows[[length(rows) + 1L]] <- row
  }
  rows
}

#' Cosine and sine Fourier pairs, exactly periodic with \code{period}
#'
#' @param t Numeric vector of times.
#' @param period Positive period.
#' @param order Integer, number of harmonics.
#' @return A list of rows, each \code{2 * order} long.
#' @export
fourier_terms <- function(t, period, order) {
  if (period <= 0)
    stop(sprintf("prphet: period must be positive, got %r", period))
  order <- as.integer(order)
  if (order < 1L)
    stop(sprintf("prphet: order must be at least 1, got %d", order))
  rows <- list()
  for (tv in t) {
    row <- numeric(0)
    for (n in seq_len(order)) {
      ang <- 2.0 * pi * n * tv / period
      row <- c(row, cos(ang), sin(ang))
    }
    rows[[length(rows) + 1L]] <- row
  }
  rows
}

#' One indicator per holiday, optionally widened by a window
#'
#' @param t Numeric vector of times.
#' @param holidays Named list of dates per holiday.
#' @param lower Integer, days before the holiday to flag.
#' @param upper Integer, days after the holiday to flag.
#' @return A list with \code{matrix} (list of rows) and \code{names}
#'   (sorted holiday names).
#' @export
holiday_matrix <- function(t, holidays, lower = 0, upper = 0) {
  names_ <- sort(names(holidays))
  rows <- list()
  for (tv in t) {
    row <- numeric(0)
    for (nm in names_) {
      hit <- 0.0
      for (d in holidays[[nm]]) {
        if (d - lower <= tv && tv <= d + upper) {
          hit <- 1.0
          break
        }
      }
      row <- c(row, hit)
    }
    rows[[length(rows) + 1L]] <- row
  }
  list(matrix = rows, names = names_)
}

#' Stack trend, seasonality and holiday columns into one design
#'
#' @param t Numeric vector of times.
#' @param cps Numeric vector of changepoints.
#' @param seasonalities Optional list of \code{c(name, period, order)}.
#' @param holidays Optional named list of dates per holiday.
#' @param holiday_window \code{c(lower, upper)} window around each date.
#' @return A list with \code{X} (list of rows), \code{cols}, \code{hn}.
#' @export
prophet_design <- function(t, cps, seasonalities = NULL, holidays = NULL,
                           holiday_window = c(0, 0)) {
  tm <- trend_matrix(t, cps)
  cols <- c("k", "m", paste0("delta_", seq_along(cps) - 1L))
  blocks <- list(tm)
  seas <- seasonalities
  if (!is.null(seas)) {
    for (s in seas) {
      blocks[[length(blocks) + 1L]] <-
        fourier_terms(t, s[[2L]], s[[3L]])
      for (n in seq_len(as.integer(s[[3L]]))) {
        cols <- c(cols, paste0(s[[1L]], "_cos", n),
                  paste0(s[[1L]], "_sin", n))
      }
    }
  }
  hn <- character(0)
  if (!is.null(holidays)) {
    hm <- holiday_matrix(t, holidays, holiday_window[1L],
                         holiday_window[2L])
    blocks[[length(blocks) + 1L]] <- hm$matrix
    hn <- hm$names
    cols <- c(cols, paste0("holiday_", hn))
  }
  X <- lapply(seq_along(t), function(i)
    unlist(lapply(blocks, function(b) b[[i]])))
  list(X = X, cols = cols, hn = hn)
}

#' Fit a Prophet model by penalised least squares
#'
#' Soft-thresholded cyclic coordinate descent. The Laplace prior on the
#' deltas is an L1 penalty of \code{1/changepoint_prior} on the rate
#' adjustments only -- never on k, m, the seasonal coefficients or the
#' holiday indicators.
#'
#' @param t Numeric vector of times.
#' @param y Numeric vector of observations.
#' @param n_changepoints Integer number of candidate changepoints.
#' @param changepoint_range Fraction of the time range to span.
#' @param changepoints Optional explicit vector of changepoints.
#' @param seasonalities Optional list of \code{c(name, period, order)}.
#' @param holidays Optional named list of dates per holiday.
#' @param holiday_window \code{c(lower, upper)} window around each date.
#' @param changepoint_prior Positive Laplace scale tau.
#' @param ridge Small L2 stabiliser on every coefficient.
#' @return A list with \code{estimate}, \code{fitted}, \code{residual},
#'   \code{coef}, \code{beta}, \code{columns}, \code{changepoints},
#'   \code{deltas}, \code{k}, \code{m}, \code{trend}, \code{holiday_names},
#'   \code{t}, \code{n}, \code{changepoint_prior},
#'   \code{n_active_changepoints}, \code{sigma}, \code{seasonalities},
#'   \code{method}.
#' @export
morie_prphet <- function(t, y, n_changepoints = 10L, changepoint_range = 0.8,
                         changepoints = NULL, seasonalities = NULL,
                         holidays = NULL, holiday_window = c(0, 0),
                         changepoint_prior = 0.05, ridge = 1e-8) {
  tv <- as.numeric(t)
  yv <- as.numeric(y)
  n <- length(tv)
  if (length(yv) != n)
    stop(sprintf("prphet: %d times but %d observations", n, length(yv)))
  if (n < 8L)
    stop(sprintf("prphet: need at least 8 observations, got %d", n))
  tau <- as.numeric(changepoint_prior)
  if (tau <= 0)
    stop(sprintf("prphet: changepoint_prior must be positive, got %r",
                 changepoint_prior))
  cps <- .changepoints(tv, as.integer(n_changepoints), changepoint_range,
                       changepoints)
  ds <- prophet_design(tv, cps, seasonalities, holidays, holiday_window)
  X <- ds$X
  cols <- ds$cols
  hn <- ds$hn
  p <- length(cols)
  pen <- rep(0.0, p)
  for (j in seq_along(cols))
    if (substr(cols[j], 1L, 6L) == "delta_") pen[j] <- 1.0 / tau
  # XtX and Xty
  XtX <- matrix(0, p, p)
  for (a in seq_len(p)) for (b in seq_len(p)) {
    s <- 0.0
    for (i in seq_len(n)) s <- s + X[[i]][a] * X[[i]][b]
    XtX[a, b] <- s
  }
  Xty <- numeric(p)
  for (a in seq_len(p)) {
    s <- 0.0
    for (i in seq_len(n)) s <- s + X[[i]][a] * yv[i]
    Xty[a] <- s
  }
  beta <- rep(0.0, p)
  for (iter in seq_len(400L)) {
    shift <- 0.0
    for (a in seq_len(p)) {
      gaa <- XtX[a, a] + ridge
      if (gaa <= 0) next
      r <- Xty[a] - sum(XtX[a, -a] * beta[-a])
      if (pen[a] > 0) {
        nb <- if (abs(r) <= pen[a]) 0.0
              else (r - sign(r) * pen[a]) / gaa
      } else {
        nb <- r / gaa
      }
      if (abs(nb - beta[a]) > shift) shift <- abs(nb - beta[a])
      beta[a] <- nb
    }
    if (shift < 1e-12) break
  }
  fitted <- numeric(n)
  for (i in seq_len(n)) {
    s <- 0.0
    for (a in seq_len(p)) s <- s + X[[i]][a] * beta[a]
    fitted[i] <- s
  }
  resid <- yv - fitted
  named <- as.list(setNames(beta, cols))
  deltas <- numeric(length(cps))
  for (j in seq_along(cps))
    deltas[j] <- named[[paste0("delta_", j - 1L)]]
  list(estimate = fitted, fitted = fitted, residual = resid,
       coef = named, beta = beta, columns = cols,
       changepoints = cps, deltas = deltas,
       k = named$k, m = named$m,
       trend = piecewise_trend(tv, named$k, named$m, deltas, cps),
       holiday_names = hn, t = tv, n = n,
       changepoint_prior = tau,
       n_active_changepoints = sum(deltas != 0),
       sigma = sqrt(sum(resid^2) / max(n - p, 1L)),
       seasonalities = if (is.null(seasonalities)) character(0)
                       else vapply(seasonalities, `[[`, character(1), 1L),
       method = paste0("Prophet decomposable model, ",
                       "Taylor & Letham (2018) eq. (1) and (4)"))
}

#' Forecast at new times, reusing the fitted coefficients
#'
#' @param fit A fit object as returned by \code{\link{morie_prphet}}.
#' @param t_new Numeric vector of new times.
#' @param seasonalities Same as for the fit.
#' @param holidays Same as for the fit.
#' @param holiday_window Same as for the fit.
#' @return Numeric vector of forecasts.
#' @export
prophet_predict <- function(fit, t_new, seasonalities = NULL,
                            holidays = NULL, holiday_window = c(0, 0)) {
  tn <- as.numeric(t_new)
  ds <- prophet_design(tn, fit$changepoints, seasonalities, holidays,
                       holiday_window)
  if (!identical(ds$cols, fit$columns))
    stop("prphet: the prediction design does not match the fitted one; ",
         "pass the same seasonalities and holidays")
  X <- ds$X
  beta <- fit$beta
  vapply(seq_along(tn), function(i)
    sum(X[[i]] * beta), numeric(1))
}

#' Compact alias per ledger/NAMING.md
#' @export
#' @noRd
prophetfit <- morie_prphet

#' Public alias resolved by fn/_lazy_map.json
#' @export
#' @noRd
prophet <- morie_prphet

#' @export
prphet_cheatsheet <- function() {
  paste0("prphet: y = g(t) + s(t) + h(t) + eps. Trend g = (k + ",
         "a(t)'delta)t + (m + a(t)'gamma) with gamma_j = -s_j ",
         "delta_j -- that is what JOINS the segments; without it the ",
         "curve jumps at every changepoint and least squares hides ",
         "it in the residual. s(t) is a Fourier series, exactly ",
         "periodic. Holidays need their own indicators because they ",
         "move. Penalise the deltas ONLY.")
}

# Namespaced aliases so both trees expose one API. The morie tree
# defined these unprefixed, which is a collision hazard in R's single
# sourced environment, and left prnFil calling a morie_prphet_fit that
# existed only in rmorie.
#' @rdname morie_prphet
#' @export
morie_prphet_fit <- morie_prphet

#' @rdname piecewise_trend
#' @export
morie_prphet_piecewise_trend <- piecewise_trend

#' @rdname trend_matrix
#' @export
morie_prphet_trend_matrix <- trend_matrix

#' @rdname fourier_terms
#' @export
morie_prphet_fourier_terms <- fourier_terms

#' @rdname holiday_matrix
#' @export
morie_prphet_holiday_matrix <- holiday_matrix

#' @rdname prophet_design
#' @export
morie_prphet_design <- prophet_design

#' @rdname prophet_predict
#' @export
morie_prphet_predict <- prophet_predict
