# SPDX-License-Identifier: AGPL-3.0-or-later
#' Random-walk transition matrix and random-walk Laplacian.
#'
#' Formula: W = D^-1 A;  L_rw = I - W
#'
#' @param Adj Symmetric adjacency matrix with a zero diagonal; entries may be 0/1 or non-negative edge weights.
#'
#' @return List with ``W``, ``L_rw``, ``degrees``, ``stationary``, ``isolated``, ``n``.
#' @references Chung, F. (1997), Spectral Graph Theory, CBMS Regional Conference Series in Mathematics 92, American Mathematical Society, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The conventions below were taken instead from the author's own survey, Chung, F., Four proofs for the Cheeger inequality and graph partition algorithms, Proceedings of ICCM 2007 Vol. II pp. 1-4, Sect. 2 (Preliminaries), which restates them in her notation; that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  Sect. 2: the transition probability matrix is W = D^-1 A, which is not symmetric, and pi = (d_1/vol(G), ..., d_n/vol(G)) is the stationary distribution of the walk on a connected non-bipartite graph.  L_rw = I - W is similar to the normalized Laplacian through L = D^1/2 (I - W) D^-1/2, so the two have the same eigenvalues.  Isolated vertices get an all-zero row of W.
#' @export
Rwlaplac <- function(Adj) {
  A <- .t1_mat(Adj); n <- nrow(A)
  if (n == 0L || ncol(A) != n) stop("Adj must be a non-empty square matrix")
  if (any(A < 0)) stop("edge weights must be non-negative")
  if (any(A != t(A))) stop("Adj must be symmetric")
  d <- rowSums(A)

  iso <- which(d == 0)
  inv <- ifelse(d == 0, 0, 1 / d)
  Wm <- inv * A
  vol <- sum(d)
  .t1_result(W = Wm, L_rw = diag(n) - Wm, degrees = d,
             stationary = if (vol == 0) numeric(n) else d / vol,
             isolated = iso, n = n,
             method = "Random-walk matrix D^-1 A and Laplacian I - W")
}
