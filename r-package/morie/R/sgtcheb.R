# SPDX-License-Identifier: AGPL-3.0-or-later
#' Cheeger ratio of a vertex subset and the Cheeger bounds on the spectral gap.
#'
#' Formula: h_S = |dS| / min{vol(S), vol(G) - vol(S)};  2 h >= lambda_G >= h^2 / 2
#'
#' @param Adj Symmetric adjacency matrix with a zero diagonal; entries may be 0/1 or non-negative edge weights.
#' @param S 1-based indices of the vertices in the subset; must be a proper non-empty subset.
#'
#' @return List with ``cheeger_ratio``, ``boundary``, ``vol_S``, ``vol_complement``, ``vol_G``, ``upper_bound``, ``lower_bound``, ``n``.
#' @references Chung, F. (1997), Spectral Graph Theory, CBMS Regional Conference Series in Mathematics 92, American Mathematical Society, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The conventions below were taken instead from the author's own survey, Chung, F., Four proofs for the Cheeger inequality and graph partition algorithms, Proceedings of ICCM 2007 Vol. II pp. 1-4, Sect. 2 (Preliminaries), which restates them in her notation; that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  Sect. 2: dS = {{u, v} in E : u in S, v not in S}, h_S = |dS| / min{vol(S), vol(G) - vol(S)}, and the Cheeger constant h_G is the minimum of h_S over all subsets.  Sect. 1 states the Cheeger inequality 2 h_G >= lambda_G >= h_G^2 / 2 for a connected graph.  Only ONE subset is supplied here, so ``cheeger_ratio`` is h_S and not the constant h_G; the two bounds are reported as what h_S would give if this subset were the minimising one, and are labelled accordingly rather than presented as a proven bracket on lambda_G.  |dS| is the total weight crossing the cut, which is the edge count for an unweighted graph.
#' @export
Cheegbnd <- function(Adj, S) {
  A <- .t1_mat(Adj); n <- nrow(A)
  if (n == 0L || ncol(A) != n) stop("Adj must be a non-empty square matrix")
  if (any(A < 0)) stop("edge weights must be non-negative")
  if (any(A != t(A))) stop("Adj must be symmetric")
  d <- rowSums(A)

  idx <- sort(unique(as.integer(S)))
  if (length(idx) == 0L) stop("S must be non-empty")
  if (any(idx < 1L | idx > n)) stop("S must contain 1-based vertex indices")
  if (length(idx) == n) stop("S must be a proper subset")
  inS <- logical(n); inS[idx] <- TRUE
  cut <- sum(A[inS, !inS, drop = FALSE])
  volS <- sum(d[inS]); volG <- sum(d); volC <- volG - volS
  den <- min(volS, volC)
  if (den <= 0) stop("both sides of the cut must have positive volume")
  h <- cut / den
  .t1_result(cheeger_ratio = h, boundary = cut, vol_S = volS,
             vol_complement = volC, vol_G = volG,
             upper_bound = 2 * h, lower_bound = h^2 / 2, n = n,
             method = "Cheeger ratio and Cheeger bounds")
}
