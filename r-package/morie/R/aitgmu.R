# SPDX-License-Identifier: AGPL-3.0-or-later
#' Geometric mean of a composition.
#'
#' Formula: g(x) = (x_1 x_2 ... x_D)^(1/D) = exp( (1/D) sum_i log x_i )
#'
#' @param x Composition with strictly positive parts.
#'
#' @return List with ``geomean``, ``log_geomean``, ``D``.
#' @references Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The definitions below were taken instead from Mateu-Figueras, G., Pawlowsky-Glahn, V. and Egozcue, J. J., The normal distribution in some constrained sample spaces, arXiv:0802.2643 (published as SORT 37(1):29-56, 2013), Sect. 4.1, which prints them with equation numbers and attributes them to Aitchison (1982, 1986); that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  g(x) is the geometric mean that the centred log-ratio divides by; computed through the log-mean so it does not overflow for large D.
#' @export
Compgmean <- function(x) {
  x <- .t1_vec(x)
  if (length(x) == 0L) stop("x must be non-empty")
  if (any(x <= 0)) stop("compositions must be strictly positive")
  lg <- mean(log(x))
  .t1_result(geomean = exp(lg), log_geomean = lg, D = length(x),
             method = "Geometric mean of a composition")
}
