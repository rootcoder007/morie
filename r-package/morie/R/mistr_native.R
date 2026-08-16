# mistr -- sliding-window attention, GQA, RoPE, SwiGLU, RMSNorm
# References:
#   Jiang et al. (2023) "Mistral 7B" arXiv:2310.06825
#   Su et al. (2024) "RoFormer" Neurocomputing 568, 127063
#   Shazeer (2020) "GLU Variants" arXiv:2002.05202
#   Zhang & Sennrich (2019) "RMSNorm" arXiv:1910.07467
#   Ainslie et al. (2023) "GQA" arXiv:2305.13245
#   Beltagy et al. (2020) "Longformer" arXiv:2004.05150
# Base R only.

#' mistr_rms_norm
#'
#' A step of the mistr_native implementation. Called by \code{mistr_mistral_block}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x A vector; its length is taken.
#' @param weight Optional; may be \code{NULL}. A vector; its length is taken.
#' @param eps Numeric; combined arithmetically in the body. Defaults to \code{1e-06}.
#' @return A numeric value.
#' @export
mistr_rms_norm <- function(x, weight = NULL, eps = 1e-6) {
  d <- length(x)
  if (d == 0L) stop("mistr: empty vector")
  ms <- sum(x * x) / d
  inv <- 1 / sqrt(ms + eps)
  if (is.null(weight)) {
    return(x * inv)
  }
  if (length(weight) != d) {
    stop(sprintf("mistr: gain has %d entries for %d channels",
                 length(weight), d))
  }
  x * inv * weight
}

#' mistr_swiglu
#'
#' A step of the mistr_native implementation. Called by \code{mistr_mistral_block}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x A matrix; passed to \code{crossprod}.
#' @param W1 A matrix; passed to \code{crossprod}.
#' @param W2 A matrix; passed to \code{crossprod}.
#' @param W3 A matrix; passed to \code{crossprod}.
#' @return A vector, from \code{as.numeric}.
#' @export
mistr_swiglu <- function(x, W1, W2, W3) {
  x <- as.numeric(x)
  W1 <- as.matrix(W1)
  W3 <- as.matrix(W3)
  W2 <- as.matrix(W2)
  a <- as.numeric(crossprod(x, W1))  # length ncol(W1)
  b <- as.numeric(crossprod(x, W3))
  if (length(a) != length(b)) {
    stop("mistr: W1 and W3 must have the same width")
  }
  s <- 1 / (1 + exp(-a))
  gated <- s * a * b
  as.numeric(crossprod(gated, W2))
}

#' mistr_rope_angles
#'
#' A step of the mistr_native implementation. Called by \code{mistr_apply_rope}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param d Numeric; combined arithmetically in the body.
#' @param base Numeric; combined arithmetically in the body. Defaults to \code{10000}.
#' @return A numeric value.
#' @export
mistr_rope_angles <- function(d, base = 10000) {
  if (d %% 2L != 0L) {
    stop(sprintf("mistr: RoPE needs an even dimension, got %d", d))
  }
  base^(-(2 * seq_len(d %/% 2L) - 2) / d)
}

#' mistr_apply_rope
#'
#' A step of the mistr_native implementation. Called by \code{mistr_grouped_query_attention}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x A vector; its length is taken and its elements indexed.
#' @param pos Numeric; combined arithmetically in the body.
#' @param theta Defaults to \code{NULL}.
#' @param base Defaults to \code{10000}.
#' @return The value of \code{out}, as built in the body.
#' @export
mistr_apply_rope <- function(x, pos, theta = NULL, base = 10000) {
  d <- length(x)
  th <- if (is.null(theta)) mistr_rope_angles(d, base) else theta
  if (length(th) != d %/% 2L) {
    stop(sprintf("mistr: %d angles for %d channels", length(th), d))
  }
  out <- numeric(d)
  for (i in seq_len(d %/% 2L)) {
    ang <- pos * th[i]
    c_ <- cos(ang); s_ <- sin(ang)
    a <- x[2 * i - 1L]; b <- x[2 * i]
    out[2 * i - 1L] <- a * c_ - b * s_
    out[2 * i]     <- a * s_ + b * c_
  }
  out
}

#' mistr_sliding_window_mask
#'
#' A step of the mistr_native implementation. Called by \code{mistr_mistral_block}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param L A count; the body uses it as \code{seq_len(...)}.
#' @param window See Usage.
#' @param causal A flag; the body branches on it. Defaults to \code{TRUE}.
#' @return The value of \code{mask}, as built in the body.
#' @export
mistr_sliding_window_mask <- function(L, window, causal = TRUE) {
  if (window < 1L) {
    stop(sprintf("mistr: window must be at least 1, got %d", window))
  }
  mask <- matrix(FALSE, nrow = L, ncol = L)
  for (i in seq_len(L)) {
    for (j in seq_len(L)) {
      ok <- (j <= i || !causal) && (i - j) < window
      mask[i, j] <- ok
    }
  }
  mask
}

#' mistr_attention_span
#'
#' A step of the mistr_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param window Coerced to integer by the body, with \code{as.integer}.
#' @param n_layers Coerced to integer by the body, with \code{as.integer}.
#' @return A numeric value.
#' @export
mistr_attention_span <- function(window, n_layers) {
  as.integer(window) * as.integer(n_layers)
}

#' mistr_grouped_query_attention
#'
#' A step of the mistr_native implementation. Called by \code{mistr_mistral_block}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param Q A matrix; passed to \code{as.matrix}.
#' @param K A matrix; passed to \code{as.matrix}.
#' @param V A matrix; passed to \code{as.matrix}.
#' @param n_heads A count; the body uses it as \code{seq_len(...)}.
#' @param n_kv_heads Numeric; combined arithmetically in the body.
#' @param mask Optional; may be \code{NULL}. A matrix; indexed by row and column.
#' @param positions Optional; may be \code{NULL}. A vector; its length is taken.
#' @param base Defaults to \code{10000}.
#' @return The value of \code{out}, as built in the body.
#' @export
mistr_grouped_query_attention <- function(Q, K, V, n_heads, n_kv_heads,
                                          mask = NULL, positions = NULL,
                                          base = 10000) {
  Qm <- as.matrix(Q)
  Km <- as.matrix(K)
  Vm <- as.matrix(V)
  L <- nrow(Qm)
  if (nrow(Km) != L || nrow(Vm) != L) {
    stop("mistr: Q, K and V must have the same length")
  }
  d <- ncol(Qm)
  if (n_heads < 1L || n_kv_heads < 1L) {
    stop("mistr: need at least one head of each kind")
  }
  if (n_heads %% n_kv_heads != 0L) {
    stop(sprintf("mistr: n_heads (%d) must be a multiple of n_kv_heads (%d)",
                 n_heads, n_kv_heads))
  }
  if (d %% n_heads != 0L) {
    stop(sprintf("mistr: dimension %d is not divisible by %d heads",
                 d, n_heads))
  }
  hd <- d %/% n_heads
  dk <- ncol(Km)
  if (dk != n_kv_heads * hd) {
    stop(sprintf("mistr: K and V must be %d wide (n_kv_heads=%d times head_dim=%d), got %d",
                 n_kv_heads * hd, n_kv_heads, hd, dk))
  }
  kd <- hd
  group <- n_heads %/% n_kv_heads
  pos <- if (is.null(positions)) seq_len(L) - 1L else as.numeric(positions)
  out <- matrix(0, nrow = L, ncol = d)
  apply_rope_flag <- !is.null(positions) || is.null(positions)
  # default: rotate when positions is NULL or provided, unless explicitly FALSE
  use_rope <- apply_rope_flag
  if (!is.null(positions) && length(positions) == 1L && is.logical(positions) && !positions) {
    use_rope <- FALSE
  }
  for (h in seq_len(n_heads) - 1L) {
    g <- h %/% group
    qs <- Qm[, (h * hd + 1L):((h + 1L) * hd), drop = FALSE]
    ks <- Km[, (g * kd + 1L):((g + 1L) * kd), drop = FALSE]
    vs <- Vm[, (g * kd + 1L):((g + 1L) * kd), drop = FALSE]
    if (use_rope) {
      qs <- t(sapply(seq_len(L), function(t) mistr_apply_rope(qs[t, ], pos[t], base = base)))
      ks <- t(sapply(seq_len(L), function(t) mistr_apply_rope(ks[t, ], pos[t], base = base)))
    }
    scale <- 1 / sqrt(hd)
    for (i in seq_len(L)) {
      allowed <- if (is.null(mask)) seq_len(L) else which(mask[i, ])
      if (length(allowed) == 0L) {
        stop(sprintf("mistr: row %d may attend to nothing", i))
      }
      sc <- sapply(allowed, function(j) scale * sum(qs[i, ] * ks[j, ]))
      mx <- max(sc)
      w <- exp(sc - mx)
      tot <- sum(w)
      for (c_ in seq_len(hd)) {
        out[i, h * hd + c_] <- sum(w * vs[allowed, c_]) / tot
      }
    }
  }
  out
}

#' mistr_mistral_block
#'
#' A step of the mistr_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param X A matrix; passed to \code{as.matrix}.
#' @param Wq See Usage.
#' @param Wk See Usage.
#' @param Wv See Usage.
#' @param Wo See Usage.
#' @param W1 See Usage.
#' @param W2 See Usage.
#' @param W3 See Usage.
#' @param n_heads Carried through into a list the body builds.
#' @param n_kv_heads Numeric; combined arithmetically in the body.
#' @param window Coerced to integer by the body, with \code{as.integer}.
#' @param norm1 Defaults to \code{NULL}.
#' @param norm2 Defaults to \code{NULL}.
#' @param base Defaults to \code{10000}.
#' @return A list with \code{estimate}, \code{output}, \code{attention_mask}, \code{L}, \code{d}, \code{n_heads}, \code{n_kv_heads}, \code{window}, \code{kv_cache_entries}, \code{method}.
#' @export
mistr_mistral_block <- function(X, Wq, Wk, Wv, Wo, W1, W2, W3,
                                n_heads, n_kv_heads, window,
                                norm1 = NULL, norm2 = NULL, base = 10000) {
  Xm <- as.matrix(X)
  L <- nrow(Xm)
  d <- ncol(Xm)
  mask <- mistr_sliding_window_mask(L, window)
  proj <- function(row, Wm) as.numeric(crossprod(row, Wm))
  h <- t(sapply(seq_len(L), function(t) mistr_rms_norm(Xm[t, ], norm1)))
  if (is.null(norm1)) h <- Xm
  Q <- t(sapply(seq_len(L), function(t) proj(h[t, ], Wq)))
  K <- t(sapply(seq_len(L), function(t) proj(h[t, ], Wk)))
  V <- t(sapply(seq_len(L), function(t) proj(h[t, ], Wv)))
  a <- mistr_grouped_query_attention(Q, K, V, n_heads, n_kv_heads,
                                     mask = mask, base = base)
  a <- t(sapply(seq_len(L), function(t) proj(a[t, ], Wo)))
  x1 <- Xm + a
  h2 <- t(sapply(seq_len(L), function(t) mistr_rms_norm(x1[t, ], norm2)))
  f <- t(sapply(seq_len(L), function(t) mistr_swiglu(h2[t, ], W1, W2, W3)))
  out <- x1 + f
  list(estimate = out, output = out, attention_mask = mask,
       L = L, d = d, n_heads = n_heads, n_kv_heads = n_kv_heads,
       window = as.integer(window),
       kv_cache_entries = min(as.integer(window), L) * n_kv_heads,
       method = "Mistral decoder block: SWA + GQA + RoPE + SwiGLU + RMSNorm, Jiang et al. (2023)")
}

#' mistr_cheatsheet
#'
#' A step of the mistr_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @return A character value.
#' @export
mistr_cheatsheet <- function() {
  paste("mistr: SWA -- token i attends to (i-W, i]; span grows to k*W over k layers because attention composes. RoPE rotates pairs (2i, 2i+1) by pos*theta_i, and <R_m q, R_n k> = <R_{m-n} q, k> EXACTLY. GQA shares one kv head across n_heads/n_kv query heads. SwiGLU gates; RMSNorm is scale-invariant but NOT shift-invariant.")
}

# house entry point: the package exports one morie_<module>
morie_mistr <- mistr_rms_norm
