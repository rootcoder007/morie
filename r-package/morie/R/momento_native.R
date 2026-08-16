# momento -- masked time-series modelling across many datasets
# References:
#   Goswami et al. (2024) "MOMENT" arXiv:2402.03885
#   Devlin et al. (2019) "BERT" arXiv:1810.04805
#   Nie et al. (2023) "PatchTST" arXiv:2211.14730
# Base R only.

#' momento_harmonise
#'
#' A step of the momento_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param series_list A vector; its length is taken.
#' @param patch_len Coerced to integer by the body, with \code{as.integer}.
#' @param normalise A flag; the body branches on it. Defaults to \code{TRUE}.
#' @return A list with \code{batch}, \code{meta}, \code{n_series}, \code{n_patches}, \code{patch_len}, \code{note}.
#' @export
momento_harmonise <- function(series_list, patch_len, normalise = TRUE) {
  P <- as.integer(patch_len)
  if (P < 1L) stop("momento: patch_len must be at least 1")
  if (length(series_list) == 0L) stop("momento: no series given")
  out <- list()
  meta <- list()
  for (s in series_list) {
    M <- as.matrix(s)
    if (nrow(M) == 0L || ncol(M) == 0L) {
      stop("momento: one of the series is empty")
    }
    D <- ncol(M)
    L <- (nrow(M) %/% P) * P
    if (L < P) {
      stop(sprintf("momento: a series has %d points, fewer than one patch of %d",
                   nrow(M), P))
    }
    for (d in seq_len(D)) {
      col <- M[seq_len(L), d]
      if (normalise) {
        m <- mean(col)
        if (L > 1L) {
          sd_ <- sqrt(sum((col - m)^2) / (L - 1))
        } else {
          sd_ <- 0
        }
        if (sd_ <= 1e-12) {
          col <- rep(0, L)
        } else {
          col <- (col - m) / sd_
        }
      } else {
        m <- 0; sd_ <- 1
      }
      patches <- split(col, (seq_along(col) - 1L) %/% P)
      out[[length(out) + 1L]] <- patches
      meta[[length(meta) + 1L]] <- list(mean = m, sd = sd_, n_patches = L %/% P)
    }
  }
  n <- min(vapply(meta, function(x) x$n_patches, integer(1)))
  batch <- lapply(out, function(row) row[seq_len(n)])
  list(batch = batch, meta = meta, n_series = length(out),
       n_patches = n, patch_len = P,
       note = "each channel is its own row, so datasets with different channel counts share a batch")
}

#' momento_mask_patches
#'
#' A step of the momento_native implementation. Called by \code{momento_reconstruction_curve}.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param patches A vector; its length is taken and its elements indexed.
#' @param mask_idx Coerced to integer by the body, with \code{as.integer}.
#' @param fill A count; the body uses it as \code{rep(...)}. Defaults to \code{0}.
#' @return A list with \code{masked}, \code{mask}, \code{mask_idx}, \code{mask_rate}, \code{n_patches}.
#' @export
momento_mask_patches <- function(patches, mask_idx, fill = 0) {
  n <- length(patches)
  idx <- sort(unique(as.integer(mask_idx)))
  if (any(idx < 0L | idx >= n)) {
    stop(sprintf("momento: a mask index is outside 0..%d", n - 1L))
  }
  if (length(idx) == 0L) {
    stop("momento: nothing was masked, so there is nothing to learn from")
  }
  if (length(idx) == n) {
    stop("momento: every patch was masked, leaving no context to reconstruct from")
  }
  masked <- lapply(seq_len(n), function(i) {
    if (i %in% idx) rep(fill, length(patches[[i]])) else as.numeric(patches[[i]])
  })
  list(masked = masked,
       mask = seq_len(n) %in% idx,
       mask_idx = idx,
       mask_rate = length(idx) / n,
       n_patches = n)
}

#' momento_masked_loss
#'
#' A step of the momento_native implementation. Called by \code{momento_reconstruction_curve}.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param truth A vector; its length is taken and its elements indexed.
#' @param reconstruction A vector; its length is taken and its elements indexed.
#' @param mask A vector; its length is taken and its elements indexed.
#' @return A list with \code{mse}, \code{n_scored}, \code{scored}.
#' @export
momento_masked_loss <- function(truth, reconstruction, mask) {
  n <- length(truth)
  if (length(reconstruction) != n || length(mask) != n) {
    stop(sprintf("momento: truth, reconstruction and mask must agree in length (%d, %d, %d)",
                 n, length(reconstruction), length(mask)))
  }
  tot <- 0; cnt <- 0
  for (i in seq_len(n)) {
    if (!mask[[i]]) next
    if (length(truth[[i]]) != length(reconstruction[[i]])) {
      stop(sprintf("momento: patch %d differs in length between truth and reconstruction", i))
    }
    d <- truth[[i]] - reconstruction[[i]]
    tot <- tot + sum(d * d)
    cnt <- cnt + length(d)
  }
  if (cnt == 0L) {
    stop("momento: no position was masked, so the loss is undefined")
  }
  list(mse = tot / cnt, n_scored = cnt,
       scored = "masked positions only -- scoring the visible ones would reward copying")
}

#' momento_task_mask
#'
#' A step of the momento_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param n_patches Coerced to integer by the body, with \code{as.integer}.
#' @param task One of \code{"forecast"}, \code{"impute"}. Defaults to \code{"forecast"}.
#' @param span Coerced to integer by the body, with \code{as.integer}. Defaults to \code{1}.
#' @param start Optional; may be \code{NULL}. Coerced to integer by the body, with \code{as.integer}.
#' @return The value of \code{seq.int}.
#' @export
momento_task_mask <- function(n_patches, task = "forecast", span = 1, start = NULL) {
  n <- as.integer(n_patches)
  s <- as.integer(span)
  tasks <- c("forecast", "impute", "classify", "anomaly")
  if (!(task %in% tasks)) {
    stop(sprintf("momento: task must be one of %s, got %s",
                 paste(tasks, collapse = ", "), task))
  }
  if (!(s >= 1L && s < n)) {
    stop(sprintf("momento: the span must lie in 1..%d, got %d", n - 1L, s))
  }
  if (task == "forecast") {
    return(seq.int(n - s, n - 1L))
  }
  if (task == "impute") {
    st <- if (is.null(start)) max(1L, (n - s) %/% 2L) else as.integer(start)
    if (st + s > n) {
      stop("momento: the imputation gap runs past the end")
    }
    return(seq.int(st, st + s - 1L))
  }
  seq.int(n - s, n - 1L)
}

#' momento_reconstruction_curve
#'
#' A step of the momento_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param patches Iterated over elementwise, with \code{lapply}.
#' @param reconstructor Accepted by the signature and not used anywhere in the body.
#' @param rates See Usage.
#' @param seed Defaults to \code{0}.
#' @return A list with \code{curve}, \code{n_patches}, \code{rates}, \code{mse}.
#' @export
momento_reconstruction_curve <- function(patches, reconstructor, rates, seed = 0) {
  P <- lapply(patches, as.numeric)
  n <- length(P)
  set.seed(seed)
  out <- list()
  for (r in rates) {
    m <- max(1L, min(n - 1L, as.integer(round(r * n))))
    idx <- sample.int(n, m)
    mk <- momento_mask_patches(P, idx)
    rec <- reconstructor(mk$masked, mk$mask)
    L <- momento_masked_loss(P, rec, mk$mask)
    out[[length(out) + 1L]] <- list(rate = mk$mask_rate, mse = L$mse, n_masked = m)
  }
  list(curve = out, n_patches = n,
       rates = vapply(out, function(o) o$rate, numeric(1)),
       mse = vapply(out, function(o) o$mse, numeric(1)))
}

#' momento_cheatsheet
#'
#' A step of the momento_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @return A character value.
#' @export
momento_cheatsheet <- function() {
  paste("momento: masked time-series pretraining. Mask patches with ZEROS and reconstruct; the loss counts the MASKED positions only, since scoring visible ones rewards copying. The hard part is multi-dataset pretraining: series differ in resolution, channel count, length and amplitude, so harmonise per-series and keep channels independent. Mask rate is a real knob -- too low is interpolation, too high leaves no context. Task changes only WHERE the mask goes: tail for forecasting, interior for imputation.")
}

# house entry point: the package exports one morie_<module>
morie_momento <- momento_harmonise
