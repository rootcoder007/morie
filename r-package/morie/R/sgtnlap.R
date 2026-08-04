# SPDX-License-Identifier: AGPL-3.0-or-later
#' Normalized graph Laplacian.
#'
#' Formula: L = I - D^(-1/2) A D^(-1/2)
#'
#' @param Adj Symmetric adjacency matrix with a zero diagonal; entries may be 0/1 or non-negative edge weights.
#'
#' @return List with ``L``, ``degrees``, ``isolated``, ``n``.
#' @references Chung, F. (1997), Spectral Graph Theory, CBMS Regional Conference Series in Mathematics 92, American Mathematical Society, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The conventions below were taken instead from the author's own survey, Chung, F., Four proofs for the Cheeger inequality and graph partition algorithms, Proceedings of ICCM 2007 Vol. II pp. 1-4, Sect. 2 (Preliminaries), which restates them in her notation; that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  Sect. 2 prints L = I - D^-1/2 A D^-1/2 = D^1/2 (I - W) D^-1/2.  Isolated vertices have d_v = 0 and no D^-1/2; the convention used here is Chung's, which sets the corresponding row and column of D^-1/2 A D^-1/2 to zero, leaving a 1 on the diagonal of L.  ``isolated`` reports their 1-based indices so the caller can see the convention was applied.
#' @export
Nlaplac <- function(Adj) {
  A <- .t1_mat(Adj); n <- nrow(A)
  if (n == 0L || ncol(A) != n) stop("Adj must be a non-empty square matrix")
  if (any(A < 0)) stop("edge weights must be non-negative")
  if (any(A != t(A))) stop("Adj must be symmetric")
  d <- rowSums(A)

  iso <- which(d == 0)
  s <- ifelse(d == 0, 0, 1 / sqrt(d))
  .t1_result(L = diag(n) - (s %o% s) * A, degrees = d, isolated = iso, n = n,
             method = "Normalized Laplacian I - D^-1/2 A D^-1/2")
}
