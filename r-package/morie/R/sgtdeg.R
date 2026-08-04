# SPDX-License-Identifier: AGPL-3.0-or-later
#' Degrees, degree matrix T and volume of a weighted graph.
#'
#' Formula: d_v = sum_u w(u, v); T = diag(d_v); vol G = sum_v d_v
#'
#' @param W Symmetric non-negative weight matrix; the diagonal is the loop
#'   weight and does count towards the degree.
#' @return List with \code{degree}, \code{T}, \code{volume},
#'   \code{isolated} (one-based), \code{n}.
#' @references Chung (1997), Spectral Graph Theory, CBMS 92, Section 1.4:
#'   "the degree d_v of a vertex v is defined to be d_v = sum_u w(u, v)"
#'   and "vol G = sum_v d_v". Fetched from the author's own copy of the
#'   chapter.
#' @export
Degmat <- function(W) {
  W <- as.matrix(W)
  n <- nrow(W)
  if (ncol(W) != n) stop("W must be square")
  if (any(W < 0)) stop("weights must be non-negative")
  if (max(abs(W - t(W))) > 1e-12) stop("W must be symmetric")
  d <- rowSums(W)
  .t1_result(degree = d, T = diag(d, n), volume = sum(d),
             isolated = which(d == 0), n = n,
             method = "Degree matrix and volume")
}
