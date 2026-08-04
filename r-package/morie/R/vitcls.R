# SPDX-License-Identifier: AGPL-3.0-or-later
#' ViT [CLS] token and position embedding
#'
#' Dosovitskiy et al. (2021), \emph{An Image is Worth 16x16 Words}, ICLR 2021,
#' arXiv:2010.11929v2, Eq. (1), p. 4:
#'
#' \deqn{z_0 = [x_{class}; x_p^1 E; \ldots ; x_p^N E] + E_{pos},}{z_0 = [x_class; x_p^1 E; ...; x_p^N E] + E_pos,}
#'
#' with E_pos in R^{(N+1) x D}.  Section 3.1, p. 3: the position embeddings are
#' standard learnable 1D ones, not 2D-aware, and z_0^0 = x_class.
#'
#' @param patches N-by-D matrix of patch embeddings.
#' @param n_patches N; must match the number of rows of patches.
#' @param cls Length-D class token; NULL is the zero vector.
#' @param pos (N+1)-by-D position embedding; NULL uses a deterministic one,
#'   and a zero matrix switches it off.
#' @return list: estimate (N+1), tokens, pos, cls, n_tokens, embed_dim, n,
#'   method.
#' @keywords internal
#' @examples
#' Vitcls(matrix(c(1, 2, 3, 4), 2, byrow = TRUE), 2, pos = matrix(0, 3, 2))$tokens
#' @export
Vitcls <- function(patches, n_patches, cls = NULL, pos = NULL) {
  Xp <- .s03mat(patches)
  N <- nrow(Xp)
  if (N == 0L) stop("vit_cls_token: patches is empty")
  D <- ncol(Xp)
  if (as.integer(n_patches) != N) {
    stop("vit_cls_token: n_patches does not match the number of rows of patches")
  }
  cc <- if (is.null(cls)) numeric(D) else .s03vec(cls)
  if (length(cc) != D) stop("vit_cls_token: cls must have length embed_dim")
  E <- if (is.null(pos)) .vit_weights(N + 1L, D, 2L) else .s03mat(pos)
  if (nrow(E) != N + 1L || ncol(E) != D) {
    stop("vit_cls_token: pos must be (n_patches+1)-by-embed_dim")
  }
  z <- matrix(0, N + 1L, D)
  for (j in seq_len(D)) z[1L, j] <- cc[j] + E[1L, j]
  for (i in seq_len(N)) {
    for (j in seq_len(D)) z[i + 1L, j] <- Xp[i, j] + E[i + 1L, j]
  }
  list(estimate = as.numeric(N + 1L), tokens = z, pos = E, cls = cc,
       n_tokens = N + 1L, embed_dim = D, n = N,
       method = "ViT [CLS] token + position embedding")
}
