# SPDX-License-Identifier: AGPL-3.0-or-later
#' ViT MLP block
#'
#' MLP(x) = GELU(x W1 + b1) W2 + b2.
#'
#' Dosovitskiy et al. (2021), \emph{An Image is Worth 16x16 Words}, ICLR 2021,
#' arXiv:2010.11929v2, p. 4, immediately above Eq. (1): "The MLP contains two
#' layers with a GELU non-linearity"; the block appears as MLP(LN(z')) in
#' Eq. (3).  The paper gives no closed form for the non-linearity, so GELU is
#' the exact one of its own source, Hendrycks and Gimpel (2016), \emph{Gaussian
#' Error Linear Units (GELUs)}, arXiv:1606.08415, GELU(u) = u Phi(u), not the
#' tanh approximation.
#'
#' @param x N-by-D matrix of tokens.
#' @param hidden_dim Width of the hidden layer.
#' @param W1,b1,W2,b2 D-by-H, H, H-by-D and D parameters; NULL uses
#'   deterministic weights and zero biases.
#' @return list: estimate (mean of the output), output, hidden, n, embed_dim,
#'   hidden_dim, method.
#' @keywords internal
#' @examples
#' Vitmlp(matrix(0, 1, 2), 3)$output
#' @export
Vitmlp <- function(x, hidden_dim, W1 = NULL, b1 = NULL, W2 = NULL, b2 = NULL) {
  X <- .s03mat(x)
  N <- nrow(X)
  if (N == 0L) stop("vit_mlp_block: x is empty")
  D <- ncol(X)
  Hd <- as.integer(hidden_dim)
  if (Hd < 1L) stop("vit_mlp_block: hidden_dim must be at least 1")
  A1 <- if (is.null(W1)) .vit_weights(D, Hd, 2L) else .s03mat(W1)
  A2 <- if (is.null(W2)) .vit_weights(Hd, D, 3L) else .s03mat(W2)
  c1 <- if (is.null(b1)) numeric(Hd) else .s03vec(b1)
  c2 <- if (is.null(b2)) numeric(D) else .s03vec(b2)
  if (nrow(A1) != D || ncol(A1) != Hd) {
    stop("vit_mlp_block: W1 must be embed_dim-by-hidden_dim")
  }
  if (nrow(A2) != Hd || ncol(A2) != D) {
    stop("vit_mlp_block: W2 must be hidden_dim-by-embed_dim")
  }
  if (length(c1) != Hd || length(c2) != D) {
    stop("vit_mlp_block: bias lengths do not match the layer widths")
  }
  pre <- .s03matmul(X, A1)
  hid <- matrix(0, N, Hd)
  for (i in seq_len(N)) {
    for (j in seq_len(Hd)) hid[i, j] <- .s03gelu(pre[i, j] + c1[j])
  }
  out <- .s03matmul(hid, A2)
  for (i in seq_len(N)) {
    for (j in seq_len(D)) out[i, j] <- out[i, j] + c2[j]
  }
  list(estimate = sum(out) / (N * D), output = out, hidden = hid,
       n = N, embed_dim = D, hidden_dim = Hd, method = "ViT MLP block")
}
