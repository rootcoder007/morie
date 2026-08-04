# SPDX-License-Identifier: AGPL-3.0-or-later
#' Cheeger constant by a sweep over the Fiedler vector.
#'
#' Formula: h(G) = min_S |boundary(S)| / min(vol S, vol S^c), searched over prefixes of the Fiedler ordering
#'
#' @param A Symmetric non-negative adjacency matrix.

#' @return List with ``sweep_min``, ``lower_bound`` (lambda_2/2), ``lambda2``, ``cut_set``, ``fiedler``, ``n``.
#' @references Cheeger (1970), A lower bound for the smallest eigenvalue of the Laplacian, in Problems in Analysis; Chung (1997), Spectral Graph Theory, AMS. Neither is held locally; the conductance definition and the sweep-cut construction are standard published results. The sweep value is checked against exhaustive enumeration over all subsets in the batch's anchor file.
#' @export
Cheeger <- function(A) {
  A <- as.matrix(A); n <- nrow(A); diag(A) <- 0
  deg <- rowSums(A)
  L <- diag(deg) - A
  e <- .t1_eigsym(L)
  lam2 <- e$values[n - 1]
  f <- e$vectors[, n - 1]
  ord <- order(f, seq_len(n))
  total <- sum(deg); best <- Inf; bestset <- integer(0)
  for (k in seq_len(n - 1)) {
    Sset <- ord[seq_len(k)]
    cut <- sum(A[Sset, -Sset, drop = FALSE])
    vol <- sum(deg[Sset]); den <- min(vol, total - vol)
    if (den > 0) {
      val <- cut / den
      if (val < best) { best <- val; bestset <- sort(Sset) }
    }
  }
  .t1_result(sweep_min = best, lower_bound = lam2 / 2, lambda2 = lam2,
             cut_set = bestset - 1L, fiedler = f, n = n,
             method = "Cheeger constant (Fiedler sweep upper bound)")
}
