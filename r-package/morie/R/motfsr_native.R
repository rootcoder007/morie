# motfsr -- motif discovery by fitting a two-component mixture model with EM (MEME/MM)
# References:
#   Bailey & Elkan (1994) ISMB-94, 28-36.
#   Bailey & Elkan (1995) ISMB-95, 21-29.
# Base R only.

#' motfsr_alphabet_of
#'
#' A step of the motfsr_native implementation. Called by \code{motfsr_prepare}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param seqs Passed to \code{unlist}.
#' @param alphabet Optional; may be \code{NULL}. Coerced to character by the body, with
#' \code{as.character}.
#' @return A vector, from \code{sort}.
#' @export
motfsr_alphabet_of <- function(seqs, alphabet) {
  if (!is.null(alphabet)) {
    a <- as.character(alphabet)
    if (length(unique(a)) != length(a)) {
      stop("motfsr: alphabet has repeated letters")
    }
    return(a)
  }
  seen <- unlist(strsplit(unlist(seqs), ""))
  if (length(seen) == 0L) stop("motfsr: the sequences contain no letters")
  sort(unique(seen))
}

#' motfsr_prepare
#'
#' A step of the motfsr_native implementation. Called by \code{motfsr_mm_fit}, \code{motfsr_run}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param sequences Coerced to character by the body, with \code{as.character}.
#' @param w Numeric; combined arithmetically in the body.
#' @param alphabet Passed to \code{motfsr_alphabet_of}.
#' @return A list with \code{coded}, \code{alpha}, \code{starts}.
#' @export
motfsr_prepare <- function(sequences, w, alphabet) {
  seqs <- as.character(sequences)
  if (length(seqs) == 0L) stop("motfsr: sequences must be non-empty")
  w <- as.integer(w)
  if (w < 1L) stop("motfsr: w (motif width) must be >= 1")
  alpha <- motfsr_alphabet_of(seqs, alphabet)
  idx <- setNames(seq_along(alpha) - 1L, alpha)
  coded <- lapply(seqs, function(s) {
    r <- integer(nchar(s))
    for (k in seq_len(nchar(s))) {
      ch <- substr(s, k, k)
      if (!ch %in% names(idx)) {
        stop(sprintf("motfsr: letter %s is not in the alphabet %s", ch, paste(alpha, collapse = "")))
      }
      r[k] <- unname(idx[ch])
    }
    r
  })
  starts <- list()
  for (i in seq_along(coded)) {
    if (length(coded[[i]]) >= w) {
      for (j in seq_len(length(coded[[i]]) - w + 1L)) {
        starts[[length(starts) + 1L]] <- c(i, j)
      }
    }
  }
  if (length(starts) == 0L) {
    stop(sprintf("motfsr: no sequence is at least w = %d long", w))
  }
  list(coded = coded, alpha = alpha, starts = starts)
}

#' motfsr_mu
#'
#' A step of the motfsr_native implementation. Called by \code{motfsr_mm_fit}, \code{motfsr_run}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param coded See Usage.
#' @param L A count; the body uses it as \code{numeric(...)}.
#' @return A numeric value.
#' @export
motfsr_mu <- function(coded, L) {
  c_ <- numeric(L)
  for (row in coded) for (k in row) c_[k + 1L] <- c_[k + 1L] + 1
  tot <- sum(c_)
  c_ / tot
}

#' motfsr_uniform_theta
#'
#' A step of the motfsr_native implementation. Called by \code{motfsr_run}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param w A count; the body uses it as \code{seq_len(...)}.
#' @param L Accepted by the signature and not used anywhere in the body.
#' @param mu Coerced to numeric by the body, with \code{as.numeric}.
#' @return The value of \code{theta}, as built in the body.
#' @export
motfsr_uniform_theta <- function(w, L, mu) {
  theta <- vector("list", w + 1L)
  theta[[1L]] <- as.numeric(mu)
  for (t in seq_len(w)) theta[[t + 1L]] <- as.numeric(mu)
  theta
}

#' motfsr_theta_from_subsequence
#'
#' A step of the motfsr_native implementation. Called by \code{motfsr_mm_fit}, \code{motfsr_run}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param coded A vector; indexed elementwise.
#' @param i See Usage.
#' @param j Numeric; combined arithmetically in the body.
#' @param w A count; the body uses it as \code{seq_len(...)}.
#' @param L A count; the body uses it as \code{rep(...)}.
#' @param mu Coerced to numeric by the body, with \code{as.numeric}.
#' @param weight Numeric; combined arithmetically in the body.
#' @return The value of \code{theta}, as built in the body.
#' @export
motfsr_theta_from_subsequence <- function(coded, i, j, w, L, mu, weight) {
  theta <- list(as.numeric(mu))
  for (t in seq_len(w)) {
    k <- coded[[i]][j + t - 1L] + 1L
    rest <- if (L > 1L) (1 - weight) / (L - 1) else 0
    row <- rep(rest, L)
    row[k] <- if (L > 1L) weight else 1
    theta[[t + 1L]] <- row
  }
  theta
}

#' motfsr_log_component
#'
#' A step of the motfsr_native implementation. Called by \code{motfsr_mm_fit}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param theta A vector; indexed elementwise.
#' @param coded A vector; indexed elementwise.
#' @param i See Usage.
#' @param j Numeric; combined arithmetically in the body.
#' @param w A count; the body uses it as \code{seq_len(...)}.
#' @param comp Passed to \code{==}.
#' @return The value of \code{tot}, as built in the body.
#' @export
motfsr_log_component <- function(theta, coded, i, j, w, comp) {
  tot <- 0
  for (t in seq_len(w)) {
    k <- coded[[i]][j + t - 1L] + 1L
    f <- if (comp == 1L) theta[[t + 1L]][k] else theta[[1L]][k]
    if (f <= 0) return(-Inf)
    tot <- tot + log(f)
  }
  tot
}

#' motfsr_normalise_windows
#'
#' A step of the motfsr_native implementation. Called by \code{motfsr_mm_fit}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param z A vector; its length is taken and its elements indexed.
#' @param w A count; the body uses it as \code{seq_len(...)}.
#' @param max_sweeps A count; the body uses it as \code{seq_len(...)}. Defaults to \code{100}.
#' @return The value of \code{z}, as built in the body.
#' @export
motfsr_normalise_windows <- function(z, w, max_sweeps = 100) {
  if (w < 2L) return(z)
  for (iter in seq_len(max_sweeps)) {
    worst <- 1 + 1e-12
    wi <- -1L
    wj <- -1L
    for (i in seq_along(z)) {
      row <- z[[i]]
      m <- length(row)
      if (m < w) {
        run <- sum(row)
        if (run > worst) { worst <- run
        wi <- i
        wj <- 0L }
        next
      }
      run <- sum(row[seq_len(w)])
      if (run > worst) { worst <- run
      wi <- i
      wj <- 0L }
      for (j in seq.int(2L, m - w + 1L)) {
        run <- run + row[j + w - 1L] - row[j - 1L]
        if (run > worst) { worst <- run
        wi <- i
        wj <- j - 1L }
      }
    }
    if (wi < 0L) break
    hi <- min(wj + w, length(z[[wi]]))
    for (j_ in (wj + 1L):hi) z[[wi]][j_] <- z[[wi]][j_] / worst
  }
  z
}

#' motfsr_mm_fit
#'
#' A step of the motfsr_native implementation. Called by \code{motfsr_run}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param sequences Passed to \code{motfsr_prepare}.
#' @param w A count; the body uses it as \code{seq_len(...)}.
#' @param alphabet Passed to \code{motfsr_prepare}.
#' @param theta0 Optional; may be \code{NULL}. Iterated over elementwise, with \code{lapply}.
#' @param lambda0 Optional; may be \code{NULL}. Coerced to numeric by the body, with
#' \code{as.numeric}.
#' @param beta Numeric; combined arithmetically in the body. Defaults to \code{0.01}.
#' @param erasing Optional; may be \code{NULL}. Iterated over elementwise, with \code{lapply}.
#' @param max_iter A count; the body uses it as \code{seq_len(...)}. Defaults to \code{1000}.
#' @param tol Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{1e-06}.
#' @param normalize_overlaps A flag; the body branches on it. Defaults to \code{TRUE}.
#' @param erase_by One of \code{"letter"}, \code{"start"}. Defaults to \code{"letter"}.
#' @return A list with \code{theta}, \code{motif}, \code{background}, \code{lambda1},
#' \code{z}, \code{log_likelihood}, \code{log_likelihood_trace}, \code{n_iter},
#' \code{converged}, \code{alphabet}, \code{w}.
#' @export
motfsr_mm_fit <- function(sequences, w, alphabet = NULL, theta0 = NULL,
                          lambda0 = NULL, beta = 0.01, erasing = NULL,
                          max_iter = 1000, tol = 1e-6,
                          normalize_overlaps = TRUE, erase_by = "letter") {
  prep <- motfsr_prepare(sequences, w, alphabet)
  coded <- prep$coded
  alpha <- prep$alpha
  starts <- prep$starts
  L <- length(alpha)
  w <- as.integer(w)
  beta <- as.numeric(beta)
  if (beta < 0) stop("motfsr: beta must be >= 0")
  tol <- as.numeric(tol)
  if (tol <= 0) stop("motfsr: tol must be > 0")
  max_iter <- as.integer(max_iter)
  if (max_iter < 1L) stop("motfsr: max_iter must be >= 1")
  n <- length(starts)
  mu <- motfsr_mu(coded, L)

  if (is.null(theta0)) {
    theta <- motfsr_theta_from_subsequence(coded, starts[[1]][1], starts[[1]][2],
                                           w, L, mu, 0.5)
  } else {
    theta <- lapply(theta0, function(r) as.numeric(r))
    if (length(theta) != w + 1L || any(sapply(theta, length) != L)) {
      stop("motfsr: theta0 must be (w + 1) x L")
    }
  }
  lam1 <- if (is.null(lambda0)) 1 / (2 * w) else as.numeric(lambda0)
  if (!(lam1 > 0 && lam1 < 1)) stop("motfsr: lambda0 must lie in (0, 1)")

  if (!(erase_by %in% c("letter", "start"))) {
    stop("motfsr: erase_by must be 'letter' or 'start'")
  }
  eps <- if (is.null(erasing)) NULL else lapply(erasing, as.numeric)

  trace <- c()
  converged <- FALSE
  it <- 0L
  z_by_seq <- NULL
  for (it in seq_len(max_iter)) {
    z_by_seq <- lapply(coded, function(row) rep(0, max(0L, length(row) - w + 1L)))
    loglik <- 0
    log_l1 <- log(lam1)
    log_l2 <- log(1 - lam1)
    for (s in seq_along(starts)) {
      i <- starts[[s]][1]
      j <- starts[[s]][2]
      a <- log_l1 + motfsr_log_component(theta, coded, i, j, w, 1L)
      b <- log_l2 + motfsr_log_component(theta, coded, i, j, w, 2L)
      m <- max(a, b)
      if (is.infinite(m) && m < 0) { z_by_seq[[i]][j] <- 0
      next }
      ea <- exp(a - m)
      eb <- exp(b - m)
      z_by_seq[[i]][j] <- ea / (ea + eb)
      loglik <- loglik + m + log(ea + eb)
    }
    trace <- c(trace, loglik)
    if (normalize_overlaps) z_by_seq <- motfsr_normalise_windows(z_by_seq, w)

    z_sum <- 0
    for (s in seq_along(starts)) {
      i <- starts[[s]][1]
      j <- starts[[s]][2]
      z_sum <- z_sum + z_by_seq[[i]][j]
    }
    lam1 <- min(max(z_sum / n, 1e-12), 1 - 1e-12)

    c_mat <- matrix(0, nrow = w + 1L, ncol = L)
    for (s in seq_along(starts)) {
      i <- starts[[s]][1]
      j <- starts[[s]][2]
      z1 <- z_by_seq[[i]][j]
      z2 <- 1 - z1
      e_start <- if (is.null(eps)) 1 else (if (erase_by == "start") eps[[i]][j] else NA)
      for (t in seq_len(w)) {
        k <- coded[[i]][j + t - 1L] + 1L
        e_letter <- if (is.null(eps)) 1 else eps[[i]][j + t - 1L]
        e <- if (is.null(eps)) 1 else (if (erase_by == "start") e_start else e_letter)
        c_mat[t + 1L, k] <- c_mat[t + 1L, k] + e * z1
        c_mat[1L, k] <- c_mat[1L, k] + z2
      }
    }
    new_theta <- vector("list", w + 1L)
    for (r in seq_len(w + 1L)) {
      denom <- sum(c_mat[r, ]) + beta
      if (denom <= 0) { new_theta[[r]] <- as.numeric(mu)
      next }
      new_theta[[r]] <- (c_mat[r, ] + beta * mu) / denom
    }
    delta <- 0
    for (r in seq_len(w + 1L)) for (k in seq_len(L)) {
      d <- new_theta[[r]][k] - theta[[r]][k]
      delta <- delta + d * d
    }
    delta <- sqrt(delta)
    theta <- new_theta
    if (delta < tol) { converged <- TRUE
    break }
  }

  list(theta = theta,
       motif = lapply(theta[-1L], as.numeric),
       background = as.numeric(theta[[1L]]),
       lambda1 = lam1,
       z = z_by_seq,
       log_likelihood = if (length(trace)) tail(trace, 1) else NaN,
       log_likelihood_trace = trace,
       n_iter = it,
       converged = converged,
       alphabet = alpha,
       w = w)
}

#' motfsr_log_odds_matrix
#'
#' A step of the motfsr_native implementation. Called by \code{motfsr_run}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param motif Iterated over elementwise, with \code{lapply}.
#' @param background A vector; indexed elementwise.
#' @return The value of \code{out}, as built in the body.
#' @export
motfsr_log_odds_matrix <- function(motif, background) {
  out <- lapply(motif, function(row) {
    sapply(seq_along(row), function(k) {
      f <- row[k]
      b <- background[k]
      if (f <= 0) -Inf else if (b <= 0) Inf else log(f / b)
    })
  })
  out
}

#' motfsr_bayes_threshold
#'
#' A step of the motfsr_native implementation. Called by \code{motfsr_run}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param lambda1 Numeric; combined arithmetically in the body.
#' @param loss Optional; may be \code{NULL}. A vector; indexed elementwise.
#' @return A numeric value.
#' @export
motfsr_bayes_threshold <- function(lambda1, loss = NULL) {
  lambda1 <- as.numeric(lambda1)
  if (!(lambda1 > 0 && lambda1 < 1)) {
    stop("motfsr: lambda1 must lie in (0, 1)")
  }
  t_ <- log((1 - lambda1) / lambda1)
  if (is.null(loss)) return(t_)
  r11 <- loss[[1]][1]
  r12 <- loss[[1]][2]
  r21 <- loss[[2]][1]
  r22 <- loss[[2]][2]
  num <- r12 - r22
  den <- r21 - r11
  if (num <= 0 || den <= 0) {
    stop("motfsr: the loss matrix must have r12 > r22 and r21 > r11 for the threshold to be defined")
  }
  t_ + log(num / den)
}

#' motfsr_score_sequence
#'
#' A step of the motfsr_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param spec A vector; its length is taken and its elements indexed.
#' @param sequence Coerced to character by the body, with \code{as.character}.
#' @param alphabet A vector; its length is taken.
#' @param threshold Optional; may be \code{NULL}. Passed to \code{is.null}.
#' @return A list with \code{scores}, \code{hits}.
#' @export
motfsr_score_sequence <- function(spec, sequence, alphabet, threshold = NULL) {
  idx <- setNames(seq_along(alphabet) - 1L, alphabet)
  w <- length(spec)
  s <- as.character(sequence)
  scores <- numeric(nchar(s) - w + 1L)
  for (j_ in seq_along(scores)) {
    tot <- 0
    for (t in seq_len(w)) {
      ch <- substr(s, j_ + t - 1L, j_ + t - 1L)
      if (!ch %in% names(idx)) {
        stop(sprintf("motfsr: letter %s is not in the alphabet", ch))
      }
      tot <- tot + spec[[t]][unname(idx[ch]) + 1L]
    }
    scores[j_] <- tot
  }
  if (is.null(threshold)) return(scores)
  hits <- which(scores >= threshold)
  list(scores = scores, hits = hits - 1L)
}

#' motfsr_lambda_grid
#'
#' A step of the motfsr_native implementation. Called by \code{motfsr_run}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param n_starts_total Numeric; combined arithmetically in the body.
#' @param n_seqs Numeric; passed to \code{sqrt}.
#' @param w Numeric; combined arithmetically in the body.
#' @param lambda0 Optional; may be \code{NULL}. Coerced to numeric by the body, with
#' \code{as.numeric}.
#' @return The value of \code{out}, as built in the body.
#' @export
motfsr_lambda_grid <- function(n_starts_total, n_seqs, w, lambda0) {
  if (!is.null(lambda0)) return(as.numeric(lambda0))
  lo <- sqrt(n_seqs) / n_starts_total
  hi <- 1 / (2 * w)
  lo <- min(max(lo, 1e-9), 0.5)
  hi <- min(max(hi, lo), 0.5)
  out <- c()
  v <- lo
  while (v < hi) { out <- c(out, v)
  v <- v * 2 }
  out <- c(out, hi)
  out
}

#' motfsr_run
#'
#' A step of the motfsr_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param sequences Passed to \code{motfsr_prepare}.
#' @param w A count; the body uses it as \code{seq_len(...)}.
#' @param alphabet Passed to \code{motfsr_prepare}.
#' @param n_motifs A count; the body uses it as \code{seq_len(...)}. Defaults to \code{1}.
#' @param beta Passed to \code{motfsr_mm_fit}. Defaults to \code{0.01}.
#' @param lambda0 Passed to \code{motfsr_lambda_grid}.
#' @param max_iter Passed to \code{motfsr_mm_fit}. Defaults to \code{1000}.
#' @param tol Passed to \code{motfsr_mm_fit}. Defaults to \code{1e-06}.
#' @param normalize_overlaps Passed to \code{motfsr_mm_fit}. Defaults to \code{TRUE}.
#' @param starts One of \code{"subsequences"}, \code{"uniform"}. Defaults to \code{"subsequences"}.
#' @param start_weight Coerced to numeric by the body, with \code{as.numeric}. Defaults
#' to \code{0.5}.
#' @param max_starts Coerced to integer by the body, with \code{as.integer}. Defaults to \code{200}.
#' @param start_scoring One of \code{"none"}, \code{"one_step"}. Defaults to \code{"one_step"}.
#' @param erase_by Passed to \code{motfsr_mm_fit}. Defaults to \code{"letter"}.
#' @param loss Passed to \code{motfsr_bayes_threshold}.
#' @return A list with \code{estimate}, \code{motifs}, \code{alphabet}, \code{w},
#' \code{n_subsequences}, \code{erasing}, \code{method}.
#' @export
motfsr_run <- function(sequences, w, alphabet = NULL, n_motifs = 1,
                       beta = 0.01, lambda0 = NULL, max_iter = 1000,
                       tol = 1e-6, normalize_overlaps = TRUE,
                       starts = "subsequences", start_weight = 0.5,
                       max_starts = 200, start_scoring = "one_step",
                       erase_by = "letter", loss = NULL) {
  if (!(starts %in% c("subsequences", "uniform"))) {
    stop("motfsr: starts must be 'subsequences' or 'uniform'")
  }
  if (!(start_scoring %in% c("one_step", "none"))) {
    stop("motfsr: start_scoring must be 'one_step' or 'none'")
  }
  n_motifs <- as.integer(n_motifs)
  if (n_motifs < 1L) stop("motfsr: n_motifs must be >= 1")
  start_weight <- as.numeric(start_weight)
  if (!(start_weight > 0 && start_weight < 1)) {
    stop("motfsr: start_weight must lie in (0, 1)")
  }
  prep <- motfsr_prepare(sequences, w, alphabet)
  coded <- prep$coded
  alpha <- prep$alpha
  all_starts <- prep$starts
  L <- length(alpha)
  w <- as.integer(w)
  mu <- motfsr_mu(coded, L)
  n <- length(all_starts)
  erasing <- lapply(coded, function(row) rep(1, length(row)))

  if (starts == "uniform") {
    cand <- list(motfsr_uniform_theta(w, L, mu))
  } else {
    step <- max(1L, as.integer(ceiling(n / max(1L, as.integer(max_starts)))))
    sel <- seq(1L, n, by = step)
    cand <- lapply(sel, function(s) {
      motfsr_theta_from_subsequence(coded, all_starts[[s]][1],
                                    all_starts[[s]][2], w, L, mu, start_weight)
    })
  }
  lam_grid <- motfsr_lambda_grid(n, length(coded), w, lambda0)

  motifs <- list()
  for (pass in seq_len(n_motifs)) {
    best <- NULL
    for (th0 in cand) {
      for (lam in lam_grid) {
        if (start_scoring == "one_step") {
          probe <- motfsr_mm_fit(sequences, w, alpha, th0, lam, beta,
                                 erasing, 1L, tol, normalize_overlaps, erase_by)
        } else {
          probe <- motfsr_mm_fit(sequences, w, alpha, th0, lam, beta,
                                 erasing, max_iter, tol, normalize_overlaps, erase_by)
        }
        key <- probe$log_likelihood
        if (is.null(best) || key > best[[1]]) {
          best <- list(key, th0, lam, probe)
        }
      }
    }
    th0 <- best[[2]]
    lam <- best[[3]]
    probe <- best[[4]]
    fit <- if (start_scoring == "none") probe else
      motfsr_mm_fit(sequences, w, alpha, th0, lam, beta, erasing,
                    max_iter, tol, normalize_overlaps, erase_by)
    spec <- motfsr_log_odds_matrix(fit$motif, fit$background)
    t_ <- motfsr_bayes_threshold(fit$lambda1, loss)
    sites <- list()
    for (i in seq_along(fit$z)) {
      for (j in seq_along(fit$z[[i]])) {
        s_ <- 0
        for (q in seq_len(w)) s_ <- s_ + spec[[q]][coded[[i]][j + q - 1L] + 1L]
        if (s_ >= t_) sites[[length(sites) + 1L]] <- c(i, j, s_)
      }
    }
    sites <- if (length(sites)) do.call(rbind, sites) else matrix(0, 0, 3)
    if (nrow(sites) > 1L) sites <- sites[order(-sites[, 3]), , drop = FALSE]
    consensus <- paste(sapply(fit$motif, function(row) {
      alpha[which.max(row)]
    }), collapse = "")
    motifs[[length(motifs) + 1L]] <- list(
      motif = fit$motif,
      background = fit$background,
      lambda1 = fit$lambda1,
      log_odds = spec,
      threshold = t_,
      sites = sites,
      n_sites = nrow(sites),
      consensus = consensus,
      log_likelihood = fit$log_likelihood,
      n_iter = fit$n_iter,
      converged = fit$converged,
      z = fit$z)

    if (pass < n_motifs) {
      z <- fit$z
      for (i in seq_along(erasing)) {
        for (j_ in seq_along(erasing[[i]])) {
          f <- 1
          lo_ <- max(1L, j_ - w + 1L)
          hi_ <- min(j_, length(z[[i]]))
          for (k in lo_:hi_) f <- f * (1 - z[[i]][k])
          erasing[[i]][j_] <- erasing[[i]][j_] * f
        }
      }
    }
  }

  list(estimate = motifs, motifs = motifs, alphabet = alpha, w = w,
       n_subsequences = n, erasing = erasing,
       method = "MM two-component mixture EM (Bailey & Elkan 1994)")
}

#' motfsr_cheatsheet
#'
#' A step of the motfsr_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @return A character value.
#' @export
motfsr_cheatsheet <- function() {
  paste("motfsr: MEME/MM motif discovery (Bailey & Elkan 1994). Break the sequences into ALL overlapping W-mers and fit a two-component mixture -- motif vs background -- by EM, so a sequence may contain zero, one or many occurrences and lambda1 estimates how often. E-step eq.4; M-step eq.5 for lambda and eq.13 for the letter frequencies, whose pseudo-counts exist because a frequency that hits 0 can never leave. z is normalised to sum to <= 1 over any W-window or EM collapses onto 'AAAAAA'. Multiple motifs come from erasing. Output is a Bayes-optimal classifier: log-odds matrix plus t = log((1 - lambda1)/lambda1).")
}

# house entry point: the package exports one morie_<module>
morie_motfsr <- motfsr_alphabet_of
