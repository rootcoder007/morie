# Chronos: forecasting by treating a series as a language.
# Source: Ansari, A. F. et al. (2024) "Chronos: Learning the Language
# of Time Series", Transactions on Machine Learning Research (10/2024),
# arXiv:2403.07815.
# Plus: Salinas, D., Flunkert, V., Gasthaus, J. & Januschowski, T.
# (2020) "DeepAR: Probabilistic forecasting with autoregressive
# recurrent networks", International Journal of Forecasting 36(3),
# 1181-1191, doi:10.1016/j.ijforecast.2019.07.001 (mean scaling).
# Plus: Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S.,
# Matena, M., Zhou, Y., Li, W. & Liu, P. J. (2020) "Exploring the
# Limits of Transfer Learning with a Unified Text-to-Text Transformer",
# JMLR 21(140), 1-67, arXiv:1910.10683 (T5 family backbone).

.CHRONOS_EPS <- 1e-12
.CHRONOS_PAD <- -1L
.CHRONOS_EOS <- -2L

#' chronos_mean_scale
#'
#' A step of the chronos_native implementation. Called by \code{chronos_tokenize}.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param x See Usage.
#' @param context Defaults to \code{NULL}.
#' @return A list with \code{scaled}, \code{scale}, \code{degenerate}, \code{context}, \code{preserves_zero}.
#' @export
chronos_mean_scale <- function(x, context = NULL) {
  v <- as.numeric(x)
  if (length(v) == 0L) stop("chronos: the series is empty")
  C <- if (is.null(context)) length(v) else as.integer(context)
  if (C < 1L || C > length(v)) {
    stop(sprintf("chronos: the context length must lie in 1..%d, got %d", length(v), C))
  }
  s <- sum(abs(v[seq_len(C)])) / C
  if (s <= .CHRONOS_EPS) {
    return(list(scaled = rep(0.0, length(v)), scale = 0.0,
                degenerate = TRUE,
                note = "the context is all zeros, so no scale is defined"))
  }
  list(scaled = v / s, scale = s, degenerate = FALSE, context = C,
       preserves_zero = TRUE)
}

#' chronos_uniform_bins
#'
#' A step of the chronos_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param lo Defaults to \code{-15}.
#' @param hi Defaults to \code{15}.
#' @param n_bins Defaults to \code{4096L}.
#' @return A list with \code{centers}, \code{edges}, \code{n_bins}, \code{scheme}, \code{range}.
#' @export
chronos_uniform_bins <- function(lo = -15.0, hi = 15.0, n_bins = 4096L) {
  B <- as.integer(n_bins)
  if (B < 2L) stop(sprintf("chronos: need at least 2 bins, got %d", B))
  if (as.numeric(hi) <= as.numeric(lo)) stop("chronos: hi must exceed lo")
  centers <- as.numeric(lo) + (as.numeric(hi) - as.numeric(lo)) * seq_len(B) - 1L / (B - 1L)
  edges <- 0.5 * (centers[seq_len(B - 1L)] + centers[seq.int(2L, B)])
  list(centers = centers, edges = edges, n_bins = B, scheme = "uniform",
       range = c(centers[1L], centers[B]))
}

#' chronos_quantile_bins
#'
#' A step of the chronos_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param samples See Usage.
#' @param n_bins Defaults to \code{4096L}.
#' @return A list with \code{centers}, \code{edges}, \code{n_bins}, \code{scheme}, \code{range}, \code{caveat}.
#' @export
chronos_quantile_bins <- function(samples, n_bins = 4096L) {
  v <- sort(as.numeric(samples))
  B <- as.integer(n_bins)
  if (length(v) < B) {
    stop(sprintf("chronos: %d samples cannot define %d quantile bins",
                 length(v), B))
  }
  centers <- v[pmin(length(v), as.integer((seq_len(B) - 0.5) * length(v) / B))]
  centers <- sort(unique(centers))
  if (length(centers) < 2L) {
    stop("chronos: the samples are too concentrated to form bins")
  }
  edges <- 0.5 * (centers[seq_len(length(centers) - 1L)] +
                   centers[seq.int(2L, length(centers))])
  list(centers = centers, edges = edges, n_bins = length(centers),
       scheme = "quantile",
       range = c(centers[1L], centers[length(centers)]),
       caveat = "fitted to the TRAINING distribution; an unseen dataset may fall where there are no bins")
}

#' chronos_quantize
#'
#' A step of the chronos_native implementation. Called by \code{chronos_tokenize}.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param x See Usage.
#' @param bins A list; the body reads \code{$centers}, \code{$edges} from it.
#' @return A list with \code{tokens}, \code{n_clipped}, \code{clipped_fraction}, \code{in_range}, \code{note}.
#' @export
chronos_quantize <- function(x, bins) {
  v <- as.numeric(x)
  c <- bins$centers
  e <- bins$edges
  out <- integer(length(v))
  clipped <- 0L
  for (k in seq_along(v)) {
    q <- v[k]
    if (q < c[1L] || q > c[length(c)]) clipped <- clipped + 1L
    j <- 0L
    while (j < length(e) && q >= e[j + 1L]) j <- j + 1L
    out[k] <- j
  }
  list(tokens = out, n_clipped = clipped,
       clipped_fraction = clipped / as.numeric(length(v)),
       in_range = clipped == 0L,
       note = "predictions are confined to [c_1, c_B]; a strong trend leaves that interval and cannot be represented")
}

#' chronos_dequantize
#'
#' A step of the chronos_native implementation. Called by \code{chronos_detokenize}.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param tokens See Usage.
#' @param bins A list; the body reads \code{$centers} from it.
#' @return The value of \code{out}, as built in the body.
#' @export
chronos_dequantize <- function(tokens, bins) {
  c <- bins$centers
  out <- numeric(0)
  for (t in tokens) {
    j <- as.integer(t)
    if (j == .CHRONOS_PAD || j == .CHRONOS_EOS) next
    if (j < 0L || j >= length(c)) {
      stop(sprintf("chronos: token %d is outside the vocabulary of %d bins",
                   j, length(c)))
    }
    out <- c(out, c[j + 1L])
  }
  out
}

#' chronos_tokenize
#'
#' A step of the chronos_native implementation. Called by \code{morie_chronos}.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param x See Usage.
#' @param bins A list; the body reads \code{$n_bins} from it.
#' @param context Defaults to \code{NULL}.
#' @param add_eos A flag; the body branches on it. Defaults to \code{TRUE}.
#' @param pad_to Defaults to \code{NULL}.
#' @return A list with \code{estimate}, \code{tokens}, \code{scale}, \code{n_clipped}, \code{clipped_fraction}, \code{vocab_size}, \code{method}, \code{ignores}.
#' @export
chronos_tokenize <- function(x, bins, context = NULL, add_eos = TRUE,
                             pad_to = NULL) {
  sc <- chronos_mean_scale(x, context = context)
  qz <- chronos_quantize(sc$scaled, bins)
  toks <- as.integer(qz$tokens)
  if (isTRUE(add_eos)) toks <- c(toks, .CHRONOS_EOS)
  if (!is.null(pad_to) && length(toks) < as.integer(pad_to)) {
    toks <- c(rep(.CHRONOS_PAD, as.integer(pad_to) - length(toks)), toks)
  }
  list(estimate = toks, tokens = toks, scale = sc$scale,
       n_clipped = qz$n_clipped,
       clipped_fraction = qz$clipped_fraction,
       vocab_size = bins$n_bins + 2L,
       method = "Chronos tokenisation: mean scaling then uniform quantisation; Ansari et al. (2024) Sec. 3.1",
       ignores = "time and frequency features, deliberately")
}

#' chronos_detokenize
#'
#' A step of the chronos_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param tokens See Usage.
#' @param bins See Usage.
#' @param scale See Usage.
#' @return A numeric value.
#' @export
chronos_detokenize <- function(tokens, bins, scale) {
  chronos_dequantize(tokens, bins) * as.numeric(scale)
}

#' chronos_forecast_summary
#'
#' A step of the chronos_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param token_probs See Usage.
#' @param bins A list; the body reads \code{$centers} from it.
#' @param quantiles Defaults to \code{c(0.1, 0.5, 0.9)}.
#' @return A list with \code{mean}, \code{quantiles}, \code{mode}, \code{note}.
#' @export
chronos_forecast_summary <- function(token_probs, bins,
                                     quantiles = c(0.1, 0.5, 0.9)) {
  p <- as.numeric(token_probs)
  c <- bins$centers
  if (length(p) != length(c)) {
    stop(sprintf("chronos: %d probabilities for %d bins", length(p), length(c)))
  }
  tot <- sum(p)
  if (tot <= .CHRONOS_EPS) stop("chronos: the predicted distribution has no mass")
  p <- p / tot
  mean <- sum(p * c)
  out <- list()
  for (qq in as.numeric(quantiles)) {
    acc <- 0.0
    pick <- c[length(c)]
    found <- FALSE
    for (i in seq_along(c)) {
      acc <- acc + p[i]
      if (acc >= qq) {
        pick <- c[i]
        found <- TRUE
        break
      }
    }
    if (!found) pick <- c[length(c)]
    out[[as.character(qq)]] <- pick
  }
  mode_idx <- which.max(p)
  list(mean = mean, quantiles = out, mode = c[mode_idx],
       note = "cross-entropy training does not know bins are ordered; the model must learn that neighbouring bins are similar")
}

#' morie_chronos
#'
#' A step of the chronos_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param x See Usage.
#' @param bins See Usage.
#' @param context Defaults to \code{NULL}.
#' @param add_eos Defaults to \code{TRUE}.
#' @param pad_to Defaults to \code{NULL}.
#' @return The value of \code{chronos_tokenize}.
#' @export
morie_chronos <- function(x, bins, context = NULL, add_eos = TRUE,
                          pad_to = NULL) {
  chronos_tokenize(x, bins, context, add_eos, pad_to)
}
