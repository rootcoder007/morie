# SPDX-License-Identifier: AGPL-3.0-or-later
#' Spectrum and spectral gap of the normalized Laplacian.
#'
#' Formula: eigenvalues of L = I - D^(-1/2) A D^(-1/2);  lambda_G = least nonzero eigenvalue
#'
#' @param Adj Symmetric adjacency matrix with a zero diagonal; entries may be 0/1 or non-negative edge weights.
#' @param tol Eigenvalues no larger than tol in absolute value count as zero when the spectral gap is picked out.
#'
#' @return List with ``eigenvalues``, ``spectral_gap``, ``n_zero``, ``lambda_max``, ``n``.
#' @references Chung, F. (1997), Spectral Graph Theory, CBMS Regional Conference Series in Mathematics 92, American Mathematical Society, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The conventions below were taken instead from the author's own survey, Chung, F., Four proofs for the Cheeger inequality and graph partition algorithms, Proceedings of ICCM 2007 Vol. II pp. 1-4, Sect. 2 (Preliminaries), which restates them in her notation; that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  Sect. 2: the spectral gap lambda_G is the least NONZERO eigenvalue of the normalized Laplacian.  Its eigenvalues lie in [0, 2] and the multiplicity of the eigenvalue 0 is the number of connected components, so ``n_zero`` counts components.  Eigenvalues are returned in increasing order.
#' @export
Glapspec <- function(Adj, tol = 1e-9) {
  A <- .t1_mat(Adj); n <- nrow(A)
  if (n == 0L || ncol(A) != n) stop("Adj must be a non-empty square matrix")
  if (any(A < 0)) stop("edge weights must be non-negative")
  if (any(A != t(A))) stop("Adj must be symmetric")
  d <- rowSums(A)

  s <- ifelse(d == 0, 0, 1 / sqrt(d))
  L <- diag(n) - (s %o% s) * A
  lam <- sort(.t1_eigsym(L)$values)
  tol <- as.numeric(tol)
  nz <- sum(abs(lam) <= tol)
  gap <- if (any(abs(lam) > tol)) lam[abs(lam) > tol][1] else NA_real_
  .t1_result(eigenvalues = lam, spectral_gap = gap, n_zero = nz,
             lambda_max = lam[length(lam)], n = n,
             method = "Normalized Laplacian spectrum and spectral gap")
}
