# SPDX-License-Identifier: AGPL-3.0-or-later
#' Log-ratio mean of a compositional data set.
#'
#' Formula: lrmean(X) = clr^-1( (1/n) sum_r clr(x_r) ) = C( g_1, ..., g_D ), g_i the geometric mean of column i
#'
#' @param X One composition per row; all parts strictly positive.
#' @param total Constant kappa the closure sums to.
#'
#' @return List with ``mean``, ``clr_mean``, ``geometric_mean``, ``total``, ``n``, ``D``.
#' @references Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read; see EXTERNAL_SOURCES.md.  The arithmetic mean of compositions is not a compositional statistic; the mean taken in log-ratio coordinates and mapped back is.  Averaging clr coordinates and inverting gives exactly the closed vector of per-part geometric means, so this is the same estimate as the compositional centre in morie.fn.aitcen -- both are returned so the identity is visible rather than assumed.  Implemented in the standard published form.  The log-ratio algebra it rests on was verified against Mateu-Figueras, Pawlowsky-Glahn and Egozcue, arXiv:0802.2643 Sect. 4.1 (fetched and archived), but this particular definition is not printed there and could not be checked against Aitchison's own text.
#' @export
Complrmean <- function(X, total = 1) {
  Xm <- .t1_mat(X); n <- nrow(Xm); D <- ncol(Xm)
  if (n == 0L) stop("X must have at least one composition")
  if (any(Xm <= 0)) stop("compositions must be strictly positive")
  L <- log(Xm)
  zbar <- colMeans(L - rowMeans(L))
  e <- exp(zbar - max(zbar)); s <- sum(e); k <- as.numeric(total)
  .t1_result(mean = k * e / s, clr_mean = zbar,
             geometric_mean = exp(colMeans(L)), total = k, n = n, D = D,
             method = "Compositional log-ratio mean")
}
