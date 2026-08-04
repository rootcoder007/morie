# SPDX-License-Identifier: AGPL-3.0-or-later
#' Variation matrix of a compositional data set.
#'
#' Formula: tau_ij = var( log(x_i / x_j) ) over the rows;  totvar = (1/(2D)) sum_i sum_j tau_ij
#'
#' @param X One composition per row; all parts strictly positive.
#'
#' @return List with ``variation``, ``total_variation``, ``n``, ``D``.
#' @references Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read; see EXTERNAL_SOURCES.md.  The variation matrix collects the sample variances of every pairwise log-ratio; it is symmetric with a zero diagonal, and its off-diagonal entries are the only compositional statistics that do not change when parts are dropped.  The total variation reported here is (1/(2D)) sum over the full matrix, equivalently (1/D) sum over i < j, which is the normalisation already in use elsewhere in this shelf (morie.fn.aitcen, morie.fn.aittvr).  Variances use the n - 1 divisor in both language arms.  Implemented in the standard published form.  The log-ratio algebra it rests on was verified against Mateu-Figueras, Pawlowsky-Glahn and Egozcue, arXiv:0802.2643 Sect. 4.1 (fetched and archived), but this particular definition is not printed there and could not be checked against Aitchison's own text.
#' @export
Compvarmat <- function(X) {
  Xm <- .t1_mat(X); n <- nrow(Xm); D <- ncol(Xm)
  if (n < 2L) stop("the variation matrix needs at least two compositions")
  if (any(Xm <= 0)) stop("compositions must be strictly positive")
  L <- log(Xm); T <- matrix(0, D, D)
  if (D > 1L) for (i in seq_len(D - 1L)) for (j in (i + 1L):D) {
    s <- stats::var(L[, i] - L[, j]); T[i, j] <- s; T[j, i] <- s
  }
  .t1_result(variation = T, total_variation = sum(T) / (2 * D), n = n, D = D,
             method = "Compositional variation matrix")
}
