# SPDX-License-Identifier: AGPL-3.0-or-later
#' Edge-weighted graph Laplacian.
#'
#' Formula: L = T - W,  T = diag(t_v) with t_v = sum_u w_uv
#'
#' @param W Symmetric matrix of non-negative edge weights with a zero diagonal.
#'
#' @return List with ``L``, ``strength``, ``T``, ``volume``, ``n``.
#' @references Chung, F. (1997), Spectral Graph Theory, CBMS Regional Conference Series in Mathematics 92, American Mathematical Society, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The conventions below were taken instead from the author's own survey, Chung, F., Four proofs for the Cheeger inequality and graph partition algorithms, Proceedings of ICCM 2007 Vol. II pp. 1-4, Sect. 2 (Preliminaries), which restates them in her notation; that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  This particular matrix is not printed in the retrieved survey and is implemented in the standard published form; it should be re-checked against Chung (1997) Chapter 1 if the book is ever added to the library.  The weighted Laplacian is the combinatorial Laplacian with the degree replaced by the vertex strength t_v, the sum of the weights of the edges at v; it reduces to D - A when every weight is 0 or 1.  Kept separate from morie.fn.sgtlap so the unweighted case cannot silently accept a weighted argument.
#' @export
Wlaplac <- function(W) {
  A <- .t1_mat(W); n <- nrow(A)
  if (n == 0L || ncol(A) != n) stop("W must be a non-empty square matrix")
  if (any(A < 0)) stop("edge weights must be non-negative")
  if (any(A != t(A))) stop("W must be symmetric")
  tt <- rowSums(A)
  .t1_result(L = diag(tt, n) - A, strength = tt, T = diag(tt, n),
             volume = sum(tt), n = n, method = "Weighted Laplacian T - W")
}
