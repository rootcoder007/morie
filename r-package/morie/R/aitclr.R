# SPDX-License-Identifier: AGPL-3.0-or-later
#' Centred log-ratio transform of a single composition.
#'
#' Formula: clr(x)_i = log( x_i / g(x) ), g(x) = (prod_j x_j)^(1/D)
#'
#' @param x Strictly positive vector of parts.
#' @return List with \code{clr}, \code{geomean}, \code{sum_clr}, \code{D}.
#' @references Aitchison (1986), The Statistical Analysis of Compositional
#'   Data, Chapter 4. Verified against the reference implementation in the
#'   CRAN package compositions 2.0-9, whose clr computes
#'   LOG - rowSums(LOG)/D on the logged parts.
#' @export
Clr <- function(x) {
  x <- .t1_vec(x)
  if (any(x <= 0)) stop("compositions must be strictly positive")
  D <- length(x)
  L <- log(x)
  lg <- sum(L) / D
  z <- L - lg
  .t1_result(clr = z, geomean = exp(lg), sum_clr = sum(z), D = D,
             method = "Centred log-ratio transform")
}
