# SPDX-License-Identifier: AGPL-3.0-or-later
#' ViT self-attention block
#'
#' Dosovitskiy et al. (2021), \emph{An Image is Worth 16x16 Words:
#' Transformers for Image Recognition at Scale}, ICLR 2021,
#' arXiv:2010.11929v2, Appendix A, p. 13:
#'
#' \deqn{A = \mathrm{softmax}(q k^T / \sqrt{D_h}),}{A = softmax(q k^T / sqrt(D_h)),}
#' \deqn{\mathrm{SA}(z) = A v.}{SA(z) = A v.}
#'
#' q, k and v are the projections of Eq. (5); this function takes them
#' already projected, which is what the module name (a block, not a whole
#' encoder) asks for.
#'
#' @param q,k Query and key matrices, D_h columns each.
#' @param v Value matrix, one row per key.
#' @param mask One row per query and one column per key; a zero entry blocks
#'   that key for that query, a non-zero entry keeps it (so a boolean
#'   keep-mask works unchanged).  NULL keeps everything.
#' @return list: estimate (mean of the output), attn (rows sum to 1), output,
#'   n, d_head, d_value, method.
#' @keywords internal
#' @examples
#' Vitatt(matrix(c(1, 0), 1), matrix(c(1, 0, 0, 1), 2, byrow = TRUE),
#'        matrix(c(1, 0), 2))$output
#' @export
Vitatt <- function(q, k, v, mask = NULL) {
  Q <- .s03mat(q); K <- .s03mat(k); V <- .s03mat(v)
  n <- nrow(Q); nk <- nrow(K)
  if (n == 0L || nk == 0L || nrow(V) == 0L) {
    stop("vit_self_attention: q, k and v must be non-empty")
  }
  if (nrow(V) != nk) {
    stop("vit_self_attention: k and v must have the same number of rows")
  }
  dh <- ncol(Q)
  if (ncol(K) != dh) stop("vit_self_attention: q and k must have the same width")
  dv <- ncol(V)
  M <- NULL
  if (!is.null(mask)) {
    M <- .s03mat(mask)
    if (nrow(M) != n || ncol(M) != nk) {
      stop("vit_self_attention: mask must be one row per query and one column per key")
    }
  }
  scale <- 1 / sqrt(dh)
  A <- matrix(0, n, nk)
  for (i in seq_len(n)) {
    s <- numeric(0); keep <- integer(0)
    for (j in seq_len(nk)) {
      if (!is.null(M) && M[i, j] == 0) next
      acc <- 0
      for (p in seq_len(dh)) acc <- acc + Q[i, p] * K[j, p]
      s <- c(s, acc * scale); keep <- c(keep, j)
    }
    if (length(keep) == 0L) stop("vit_self_attention: mask leaves a query with no keys")
    w <- .s03softmax(s)
    for (t in seq_along(keep)) A[i, keep[t]] <- w[t]
  }
  out <- .s03matmul(A, V)
  list(estimate = sum(out) / (n * dv), attn = A, output = out,
       n = n, d_head = dh, d_value = dv,
       method = "ViT self-attention block")
}

# Deterministic stand-in for a trained projection: .s03normdraws laid out row
# by row and divided by sqrt(fan-in).  The Python arm calls the same van der
# Corput / AS 241 sequence, so the two land on identical numbers.
.vit_weights <- function(r, c, base = 2L, mult = 1) {
  z <- .s03normdraws(r * c, base)
  matrix(z * (mult / sqrt(r)), nrow = r, ncol = c, byrow = TRUE)
}
