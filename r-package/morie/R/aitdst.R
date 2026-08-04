# SPDX-License-Identifier: AGPL-3.0-or-later
#' Aitchison distance between two compositions.
#'
#' Formula: d_a(x, y) = sqrt( sum_i (clr(x)_i - clr(y)_i)^2 )
#'   = sqrt( (1/D) sum_{i<j} (log(x_i/x_j) - log(y_i/y_j))^2 )
#'
#' @param x,y Strictly positive vectors of parts, the same length.
#' @return List with \code{distance}, \code{distance_pairwise}, \code{D}.
#'   The two agree; both are returned as a self-check.
#' @references Aitchison (1986), The Statistical Analysis of Compositional
#'   Data, Chapter 8. Verified against the reference implementation in the
#'   CRAN package compositions 2.0-9, whose dist on an acomp is the
#'   Euclidean distance of the clr coordinates.
#' @export
Compdist <- function(x, y) {
  x <- .t1_vec(x); y <- .t1_vec(y)
  if (length(x) != length(y)) stop("x and y must have the same number of parts")
  if (any(x <= 0) || any(y <= 0)) stop("compositions must be strictly positive")
  D <- length(x)
  Lx <- log(x); Ly <- log(y)
  zx <- Lx - sum(Lx) / D
  zy <- Ly - sum(Ly) / D
  pw <- 0
  for (i in seq_len(D)) for (j in seq_len(D)) if (j > i)
    pw <- pw + ((Lx[i] - Lx[j]) - (Ly[i] - Ly[j]))^2
  .t1_result(distance = sqrt(sum((zx - zy)^2)),
             distance_pairwise = sqrt(pw / D), D = D,
             method = "Aitchison distance")
}
