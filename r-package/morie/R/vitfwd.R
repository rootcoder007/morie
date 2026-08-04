# SPDX-License-Identifier: AGPL-3.0-or-later
#' Vision Transformer forward pass
#'
#' Dosovitskiy et al. (2021), \emph{An Image is Worth 16x16 Words}, ICLR 2021,
#' arXiv:2010.11929v2, p. 4:
#'
#' \deqn{z_0 = [x_{class}; x_p^1 E; \ldots; x_p^N E] + E_{pos},}{z_0 = [x_class; x_p^1 E; ...; x_p^N E] + E_pos,}
#' \deqn{z'_l = \mathrm{MSA}(\mathrm{LN}(z_{l-1})) + z_{l-1},}{z'_l = MSA(LN(z_{l-1})) + z_{l-1},}
#' \deqn{z_l = \mathrm{MLP}(\mathrm{LN}(z'_l)) + z'_l,}{z_l = MLP(LN(z'_l)) + z'_l,}
#' \deqn{y = \mathrm{LN}(z_L^0),}{y = LN(z_L^0),}
#'
#' and Appendix A, p. 13, Eq. (8): MSA(z) = [SA_1(z); ...; SA_k(z)] U_msa with
#' U_msa in R^{(k . D_h) x D} and D_h = D / k.
#'
#' Layer normalisation is not defined in the paper, which cites Ba, J. L.,
#' Kiros, J. R. and Hinton, G. E. (2016), \emph{Layer Normalization},
#' arXiv:1607.06450: LN(u) = (u - mean(u)) / sqrt(var(u) + eps), with the
#' population variance, unit gain and zero bias.  eps is an implementation
#' choice, not the paper's.
#'
#' All weights are deterministic (see .vit_weights) so that the R and Python
#' arms return the same numbers.  scale multiplies the attention and MLP
#' weights only; scale = 0 switches both blocks off, leaving the residual
#' stream of Eqs. (2)-(3) untouched, which is what the closed-form check uses.
#'
#' @param x H-by-W single-channel image, patch_size dividing H and W.
#' @param patch_size,embed_dim,num_heads,num_layers P, D, k and L; k must
#'   divide D.
#' @param mlp_ratio Hidden width of the MLP as a multiple of D (4 in the
#'   paper's configurations, Table 1).
#' @param scale Multiplier on the block weights; NULL means 1.
#' @param eps LN epsilon.
#' @param E,pos Patch projection and position embedding; NULL uses the
#'   deterministic ones.
#' @return list: estimate (mean of y), y, tokens, n_patches, n_tokens,
#'   embed_dim, n_heads, n_layers, n, method.
#' @keywords internal
#' @examples
#' Vitfwd(matrix(1:16, 4, 4, byrow = TRUE), 2, 4, 2, 1)$y
#' @export
Vitfwd <- function(x, patch_size, embed_dim, num_heads, num_layers,
                   mlp_ratio = 4L, scale = NULL, eps = 1e-5,
                   E = NULL, pos = NULL) {
  D <- as.integer(embed_dim); k <- as.integer(num_heads)
  L <- as.integer(num_layers); P <- as.integer(patch_size)
  if (k < 1L) stop("vit_forward: num_heads must be at least 1")
  if (L < 1L) stop("vit_forward: num_layers must be at least 1")
  if (D %% k != 0L) stop("vit_forward: num_heads must divide embed_dim")
  if (as.integer(mlp_ratio) < 1L) stop("vit_forward: mlp_ratio must be at least 1")
  mult <- if (is.null(scale)) 1 else as.numeric(scale)
  dh <- D %/% k
  Emat <- if (is.null(E)) .vit_weights(P * P, D, 2L) else .s03mat(E)
  pe <- Vitptm(x, P, D, E = Emat)
  N <- pe$n_patches
  posm <- if (is.null(pos)) .vit_weights(N + 1L, D, 3L) else .s03mat(pos)
  z <- Vitcls(pe$embeddings, N, cls = NULL, pos = posm)$tokens
  nt <- N + 1L
  Hd <- as.integer(mlp_ratio) * D
  for (l in seq_len(L)) {
    u <- .vit_ln(z, eps)
    cat_ <- matrix(0, nt, k * dh)
    for (h in seq_len(k)) {
      U <- .vit_weights(D, 3L * dh, 2L + (l - 1L) * k + (h - 1L), mult)
      proj <- .s03matmul(u, U)
      q <- proj[, seq_len(dh), drop = FALSE]
      kk <- proj[, dh + seq_len(dh), drop = FALSE]
      vv <- proj[, 2L * dh + seq_len(dh), drop = FALSE]
      sa <- Vitatt(q, kk, vv)$output
      cat_[, (h - 1L) * dh + seq_len(dh)] <- sa
    }
    Umsa <- .vit_weights(k * dh, D, 101L + l, mult)
    msa <- .s03matmul(cat_, Umsa)
    zp1 <- msa + z
    un <- .vit_ln(zp1, eps)
    mlp <- Vitmlp(un, Hd,
                  W1 = .vit_weights(D, Hd, 201L + l, mult),
                  W2 = .vit_weights(Hd, D, 301L + l, mult))$output
    z <- mlp + zp1
  }
  y <- .vit_ln(matrix(z[1L, ], nrow = 1L), eps)[1L, ]
  list(estimate = sum(y) / D, y = y, tokens = z, n_patches = N,
       n_tokens = nt, embed_dim = D, n_heads = k, n_layers = L, n = N,
       method = "Vision Transformer forward pass")
}

# LN with unit gain and zero bias, row by row (Ba, Kiros and Hinton 2016).
.vit_ln <- function(Z, eps) {
  out <- Z
  d <- ncol(Z)
  for (i in seq_len(nrow(Z))) {
    m <- sum(Z[i, ]) / d
    s <- sqrt(sum((Z[i, ] - m)^2) / d + eps)
    out[i, ] <- if (s == 0) rep(0, d) else (Z[i, ] - m) / s
  }
  out
}
