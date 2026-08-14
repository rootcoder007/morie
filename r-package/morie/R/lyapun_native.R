# Sources: Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993)
# "A practical method for calculating largest Lyapunov exponents from
# small data sets", Physica D 65(1-2), 117-134.

.as_series <- function(y) {
  out <- as.numeric(y)
  if (length(out) < 10L)
    stop(sprintf("lyapun: need at least 10 observations, got %d", length(out)))
  if (any(!is.finite(out)))
    stop("lyapun: the series contains a non-finite value")
  out
}

embed <- function(y, m, tau) {
  y <- .as_series(y)
  m <- as.integer(m); tau <- as.integer(tau)
  if (m < 1L) stop("lyapun: the embedding dimension must be >= 1")
  if (tau < 1L) stop("lyapun: the reconstruction delay must be >= 1")
  n_pts <- length(y) - (m - 1L) * tau
  if (n_pts < 3L)
    stop(sprintf("lyapun: m = %d and J = %d leave only %d reconstructed points",
                 m, tau, n_pts))
  out <- vector("list", n_pts)
  for (j in seq_len(n_pts)) {
    row <- numeric(m)
    for (k in seq_len(m)) row[k] <- y[j + (k - 1L) * tau]
    out[[j]] <- row
  }
  out
}

autocorrelation_lag <- function(y, threshold = NULL) {
  y <- .as_series(y)
  n <- length(y)
  if (is.null(threshold)) threshold <- 1.0 - 1.0 / exp(1.0)
  mu <- sum(y) / n
  c0 <- sum((y - mu) ^ 2) / n
  if (c0 <= 0.0)
    stop("lyapun: the series is constant, so no delay can be chosen from its autocorrelation")
  for (lag in seq_len(n - 1L)) {
    c <- 0.0
    for (t in seq_len(n - lag)) c <- c + (y[t] - mu) * (y[t + lag] - mu)
    c <- c / n
    if (c / c0 <= threshold) return(lag)
  }
  1L
}

mean_period <- function(y, dt = 1.0) {
  y <- .as_series(y)
  n <- length(y)
  mu <- sum(y) / n
  z <- y - mu
  spec <- fft(z)
  half <- floor(n / 2L) + 1L
  spec <- spec[seq_len(half)]
  freqs <- seq_len(half) - 1L
  if (dt > 0) freqs <- freqs / (n * dt)
  power <- (Re(spec) ^ 2 + Im(spec) ^ 2)
  power[1L] <- 0
  wsum <- sum(power)
  if (wsum <= 0.0) return(1.0)
  f_mean <- sum(freqs * power) / wsum
  if (f_mean <= 0.0) return(as.numeric(n))
  (1.0 / f_mean) / dt
}

.nearest_neighbours <- function(pts, min_sep) {
  n_pts <- length(pts); m <- length(pts[[1]])
  nn <- rep(-1L, n_pts)
  d0 <- rep(0.0, n_pts)
  for (j in seq_len(n_pts)) {
    best <- -1L; best_d <- Inf
    pj <- pts[[j]]
    for (jp in seq_len(n_pts)) {
      if (abs(j - jp) <= min_sep) next
      pk <- pts[[jp]]
      s <- 0.0
      early <- FALSE
      for (k in seq_len(m)) {
        diff <- pj[k] - pk[k]
        s <- s + diff * diff
        if (s >= best_d) { early <- TRUE; break }
      }
      if (!early && s < best_d) { best_d <- s; best <- jp }
    }
    nn[j] <- best
    d0[j] <- if (best >= 0L) sqrt(best_d) else NaN
  }
  list(nn = nn, d0 = d0)
}

.distance <- function(pts, a, b) {
  s <- 0.0
  pa <- pts[[a]]; pb <- pts[[b]]
  for (k in seq_along(pa)) {
    diff <- pa[k] - pb[k]
    s <- s + diff * diff
  }
  sqrt(s)
}

divergence_curve <- function(y, m = NULL, tau = NULL, dt = 1.0,
                              min_sep = NULL, max_steps = NULL) {
  y <- .as_series(y)
  n <- length(y)
  if (is.null(tau)) tau <- autocorrelation_lag(y)
  if (is.null(m)) m <- 3L
  if (dt <= 0) stop("lyapun: the sampling period must be positive")
  pts <- embed(y, m, tau)
  n_pts <- length(pts)
  if (is.null(min_sep)) min_sep <- as.integer(round(mean_period(y, dt)))
  min_sep <- as.integer(min_sep)
  if (min_sep < 0L) stop("lyapun: min_sep must be >= 0")
  if (min_sep >= n_pts - 2L)
    stop(sprintf("lyapun: the mean period (%d samples) leaves no admissible neighbours among %d reconstructed points; pass min_sep explicitly",
                 min_sep, n_pts))
  nn_d0 <- .nearest_neighbours(pts, min_sep)
  nn <- nn_d0$nn; d0 <- nn_d0$d0
  usable <- which(nn >= 0L & d0 > 0.0)
  if (length(usable) < 3L)
    stop("lyapun: fewer than three usable neighbour pairs")
  if (is.null(max_steps)) max_steps <- max(1L, n_pts %/% 4L)
  max_steps <- as.integer(max_steps)

  times <- numeric(0); curve <- numeric(0); ratio <- numeric(0); counts <- integer(0)
  for (i in seq_len(max_steps + 1L) - 1L) {
    tot <- 0.0; tot_ratio <- 0.0; cnt <- 0L
    for (j in usable) {
      jp <- nn[j]
      if (j + i > n_pts || jp + i > n_pts) next
      d <- .distance(pts, j + i, jp + i)
      if (d <= 0.0) next
      tot <- tot + log(d)
      tot_ratio <- tot_ratio + log(d / d0[j])
      cnt <- cnt + 1L
    }
    if (cnt == 0L) break
    times <- c(times, i * dt)
    curve <- c(curve, tot / cnt)
    ratio <- c(ratio, tot_ratio / cnt)
    counts <- c(counts, cnt)
  }
  list(time = times, log_divergence = curve, log_ratio = ratio,
       n_pairs = counts, neighbour = nn, d0 = d0, points = pts,
       m = m, tau = tau, min_sep = min_sep, n_points = n_pts, n_obs = n)
}

.linear_region <- function(curve, lo_frac = 0.1, hi_frac = 0.8) {
  n <- length(curve)
  if (n < 4L) return(c(0L, n))
  c_lo <- min(curve); c_hi <- max(curve)
  span <- c_hi - c_lo
  if (span <= 0.0) return(c(0L, n))
  top <- which.max(curve)
  if (top < 3L) return(c(0L, n))
  lo_level <- c_lo + lo_frac * span
  hi_level <- c_lo + hi_frac * span
  lo <- 0L
  while (lo < top && curve[lo + 1L] < lo_level) lo <- lo + 1L
  hi <- lo
  while (hi < top && curve[hi + 1L] < hi_level) hi <- hi + 1L
  hi <- min(hi + 1L, n)
  if (hi - lo < 3L) return(c(0L, n))
  c(lo, hi)
}

.ols_slope <- function(xs, ys) {
  n <- length(xs)
  mx <- mean(xs); my <- mean(ys)
  sxx <- sum((xs - mx) ^ 2)
  if (sxx <= 0) stop("lyapun: the fitting window has no spread in time")
  sxy <- sum((xs - mx) * (ys - my))
  slope <- sxy / sxx
  intercept <- my - slope * mx
  resid <- ys - intercept - slope * xs
  sse <- sum(resid ^ 2)
  sst <- sum((ys - my) ^ 2)
  se <- if (n > 2L && sse > 0) sqrt(sse / (n - 2L) / sxx) else 0.0
  r2 <- if (sst > 0) 1.0 - sse / sst else 1.0
  c(slope = slope, intercept = intercept, se = se, r2 = r2)
}

lyapunov_exponent <- function(y, embedding = NULL, tau = NULL, dt = 1.0,
                               fit = NULL, min_sep = NULL, max_steps = NULL,
                               method = "rosenstein", k = NULL) {
  if (!(method %in% c("rosenstein", "sato", "sato_k")))
    stop("lyapun: method must be 'rosenstein', 'sato' or 'sato_k'")
  dv <- divergence_curve(y, m = embedding, tau = tau, dt = dt,
                          min_sep = min_sep, max_steps = max_steps)
  times <- dv$time; curve <- dv$log_divergence
  n_steps <- length(curve)
  if (is.null(fit)) {
    lr <- .linear_region(curve); lo <- lr[1]; hi <- lr[2]
  } else {
    lo <- as.integer(fit[1]); hi <- as.integer(fit[2])
    if (lo < 0L || hi > n_steps || hi - lo < 2L)
      stop(sprintf("lyapun: the fitting window must lie inside 0..%d and span at least two steps",
                   n_steps))
  }
  ols <- .ols_slope(times[(lo + 1L):hi], curve[(lo + 1L):hi])
  slope <- ols["slope"]; intercept <- ols["intercept"]
  se <- ols["se"]; r2 <- ols["r2"]

  i_end <- hi - 1L
  sato <- if (i_end > 0L) dv$log_ratio[i_end] / (i_end * dt) else NaN

  if (is.null(k)) k <- max(1L, hi - lo)
  k <- as.integer(k)
  sato_k <- NaN; sato_k_curve <- numeric(0)
  if (k >= 1L && n_steps > k) {
    pts <- dv$points; nn <- dv$neighbour; n_pts <- dv$n_points
    usable <- which(nn >= 0L & dv$d0 > 0)
    for (i in seq_len(n_steps - k) - 1L) {
      tot <- 0.0; cnt <- 0L
      for (j in usable) {
        jp <- nn[j]
        if (max(j, jp) + i + k > n_pts) next
        d_i <- .distance(pts, j + i, jp + i)
        d_ik <- .distance(pts, j + i + k, jp + i + k)
        if (d_i <= 0.0 || d_ik <= 0.0) next
        tot <- tot + log(d_ik / d_i)
        cnt <- cnt + 1L
      }
      if (cnt == 0L) break
      sato_k_curve <- c(sato_k_curve, tot / cnt / (k * dt))
    }
    if (length(sato_k_curve) > 0L) {
      search_len <- max(3L, min(hi, length(sato_k_curve)))
      search <- sato_k_curve[seq_len(search_len)]
      w <- max(2L, length(search) %/% 4L)
      best <- 0L; best_var <- Inf
      for (s in seq_len(length(search) - w + 1L) - 1L) {
        seg <- search[(s + 1L):(s + w)]
        mu <- mean(seg)
        var <- mean((seg - mu) ^ 2)
        if (var < best_var) { best_var <- var; best <- s }
      }
      seg <- search[(best + 1L):(best + w)]
      sato_k <- mean(seg)
    }
  }

  estimate <- switch(method, rosenstein = slope, sato = sato, sato_k = sato_k)
  list(estimate = estimate, lambda1 = estimate,
       rosenstein = slope, sato = sato, sato_k = sato_k,
       sato_k_curve = sato_k_curve, se = se, r_squared = r2,
       intercept = intercept, time = times, log_divergence = curve,
       log_ratio = dv$log_ratio, n_pairs = dv$n_pairs,
       fit_range = c(lo, hi), k = k, m = dv$m, tau = dv$tau,
       min_sep = dv$min_sep, n_points = dv$n_points, n = dv$n_obs,
       dt = dt,
       method = sprintf("largest Lyapunov exponent, Rosenstein, Collins & De Luca (1993), route '%s'", method),
       note = "the exponent is the slope of <ln d_j(i)> over the initial rise; a positive value indicates chaos, and the fitting window is the caller's to choose because the curve saturates once the neighbours are as far apart as the attractor allows")
}

largest_lyapunov <- lyapunov_exponent

cheatsheet <- function() {
  "lyapun: largest Lyapunov exponent (Rosenstein, Collins & De Luca 1993). Embed with delay J and dimension m, find each point's nearest neighbour at least a mean period away, and take lambda_1 as the slope of <ln d_j(i)> against i*dt over the initial rise -- no normalisation by d_j(0) is needed, since a constant offset does not change a slope. Expected values from the paper's table 1: 0.693 for the logistic map at mu = 4, 0.418 for the Henon map. Routes: 'rosenstein' (eq. 13, default), 'sato' (eq. 9), 'sato_k' (eq. 10, whose plateau the paper itself calls unreliable)."
}

morie_lyapun <- function(y, embedding = NULL, tau = NULL, dt = 1.0,
                        fit = NULL, min_sep = NULL, max_steps = NULL,
                        method = "rosenstein", k = NULL) {
  lyapunov_exponent(y, embedding = embedding, tau = tau, dt = dt,
                    fit = fit, min_sep = min_sep, max_steps = max_steps,
                    method = method, k = k)
}
