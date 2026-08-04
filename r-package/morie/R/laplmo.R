# SPDX-License-Identifier: AGPL-3.0-or-later
#' Eigendecomposition of a graph Laplacian, with the Fiedler vector.
#'
#' Formula: L z = lambda z, eigenvalues in increasing order; the Fiedler vector is the eigenvector of the second smallest eigenvalue
#'
#' @param Adj Symmetric adjacency matrix with a zero diagonal; entries may be 0/1 or non-negative edge weights.
#' @param normalized Decompose the normalized Laplacian I - D^-1/2 A D^-1/2 rather than the combinatorial D - A.
#'
#' @return List with ``eigenvalues``, ``eigenvectors``, ``fiedler_value``, ``fiedler_vector``, ``n``.
#' @references Chung, F. (1997), Spectral Graph Theory, CBMS Regional Conference Series in Mathematics 92, American Mathematical Society, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The conventions below were taken instead from the author's own survey, Chung, F., Four proofs for the Cheeger inequality and graph partition algorithms, Proceedings of ICCM 2007 Vol. II pp. 1-4, Sect. 2 (Preliminaries), which restates them in her notation; that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  Sect. 2 for the normalized Laplacian; This particular matrix is not printed in the retrieved survey and is implemented in the standard published form; it should be re-checked against Chung (1997) Chapter 1 if the book is ever added to the library.  Eigenvalues are returned in increasing order and each eigenvector is sign-fixed so its largest-magnitude entry is positive, which is what makes the two language arms agree.  A repeated eigenvalue leaves its eigenvectors determined only up to a rotation within the eigenspace, so on a graph with a repeated Fiedler value the vector is not a stable quantity in either language.
#' @export
Glapeig <- function(Adj, normalized = TRUE) {
  A <- .t1_mat(Adj); n <- nrow(A)
  if (n == 0L || ncol(A) != n) stop("Adj must be a non-empty square matrix")
  if (any(A < 0)) stop("edge weights must be non-negative")
  if (any(A != t(A))) stop("Adj must be symmetric")
  d <- rowSums(A)

  if (isTRUE(normalized)) {
    s <- ifelse(d == 0, 0, 1 / sqrt(d))
    L <- diag(n) - (s %o% s) * A
  } else {
    L <- diag(d, n) - A
  }
  if (n < 2L) stop("the Fiedler vector needs at least two vertices")
  e <- .t1_eigsym(L)
  ord <- rev(seq_len(n))
  lam <- e$values[ord]; V <- e$vectors[, ord, drop = FALSE]
  .t1_result(eigenvalues = lam, eigenvectors = V, fiedler_value = lam[2],
             fiedler_vector = V[, 2], n = n,
             method = "Laplacian eigendecomposition with Fiedler vector")
}
