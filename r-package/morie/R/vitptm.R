# SPDX-License-Identifier: AGPL-3.0-or-later
#' ViT patch embedding
#'
#' Dosovitskiy et al. (2021), \emph{An Image is Worth 16x16 Words}, ICLR 2021,
#' arXiv:2010.11929v2, Section 3.1, p. 3 and Eq. (1), p. 4: the image
#' x in R^{H x W x C} is reshaped into a sequence of flattened 2D patches
#' x_p in R^{N x (P^2 . C)} with N = H W / P^2, and mapped to D dimensions by
#' a trainable linear projection E in R^{(P^2 . C) x D}.  The paper calls this
#' a linear projection; it is the stride-P convolution of the module name only
#' in the sense that a stride-P conv with a P-by-P kernel \emph{is} that
#' projection.
#'
#' This arm takes a single-channel image (C = 1), so the patch dimension is
#' P^2.  Patches are emitted in row-major order over the patch grid, and each
#' patch is flattened row-major.
#'
#' @param image H-by-W matrix, with patch_size dividing both H and W.
#' @param patch_size P.
#' @param embed_dim D.
#' @param E (P^2)-by-D projection; NULL uses a deterministic one.
#' @return list: estimate (N), patches, embeddings, n_patches, patch_dim,
#'   embed_dim, n, method.
#' @keywords internal
#' @examples
#' Vitptm(matrix(1:16, 4, 4, byrow = TRUE), 2, 4, E = diag(4))$patches
#' @export
Vitptm <- function(image, patch_size, embed_dim, E = NULL) {
  X <- .s03mat(image)
  H <- nrow(X)
  if (H == 0L) stop("vit_patch_embed: image is empty")
  W <- ncol(X)
  P <- as.integer(patch_size); D <- as.integer(embed_dim)
  if (P < 1L) stop("vit_patch_embed: patch_size must be at least 1")
  if (D < 1L) stop("vit_patch_embed: embed_dim must be at least 1")
  if (H %% P != 0L || W %% P != 0L) {
    stop("vit_patch_embed: patch_size must divide both image dimensions")
  }
  pd <- P * P
  N <- (H %/% P) * (W %/% P)
  patches <- matrix(0, N, pd)
  idx <- 0L
  for (gi in seq_len(H %/% P)) {
    for (gj in seq_len(W %/% P)) {
      idx <- idx + 1L
      t <- 0L
      for (a in seq_len(P)) {
        for (b in seq_len(P)) {
          t <- t + 1L
          patches[idx, t] <- X[(gi - 1L) * P + a, (gj - 1L) * P + b]
        }
      }
    }
  }
  Emat <- if (is.null(E)) .vit_weights(pd, D, 2L) else .s03mat(E)
  if (nrow(Emat) != pd || ncol(Emat) != D) {
    stop("vit_patch_embed: E must be (patch_size^2)-by-embed_dim")
  }
  Z <- .s03matmul(patches, Emat)
  list(estimate = as.numeric(N), patches = patches, embeddings = Z,
       n_patches = N, patch_dim = pd, embed_dim = D, n = N,
       method = "ViT patch embedding via 2D conv")
}
