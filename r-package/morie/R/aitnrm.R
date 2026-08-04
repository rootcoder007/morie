# SPDX-License-Identifier: AGPL-3.0-or-later
#' Aitchison norm: the distance from a composition to the barycentre.
#'
#' Formula: ||x||_a = sqrt( sum_i clr(x)_i^2 )
#'                  = sqrt( (1/D) sum_{i<j} log(x_i/x_j)^2 )
#'
#' @param x Strictly positive vector of parts.
#' @return List with \code{norm}, \code{norm_pairwise}, \code{D}. The two
#'   agree; both are returned because their agreement is the cheapest
#'   check that the clr bookkeeping is right.
#' @references Aitchison (1986), The Statistical Analysis of Compositional
#'   Data, Chapter 4. Verified against the reference implementation in the
#'   CRAN package compositions 2.0-9, whose norm on an acomp is
#'   sqrt(scalar(x, x)) with scalar the clr inner product.
#' @export
Compnorm <- function(x) {
  x <- .t1_vec(x)
  if (any(x <= 0)) stop("compositions must be strictly positive")
  D <- length(x)
  L <- log(x)
  z <- L - sum(L) / D
  pw <- 0
  for (i in seq_len(D)) for (j in seq_len(D)) if (j > i) pw <- pw + (L[i] - L[j])^2
  .t1_result(norm = sqrt(sum(z^2)), norm_pairwise = sqrt(pw / D), D = D,
             method = "Aitchison norm")
}
