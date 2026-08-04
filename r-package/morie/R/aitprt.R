# SPDX-License-Identifier: AGPL-3.0-or-later
#' Perturbation, the group operation of the simplex.
#'
#' Formula: x (+) y = C( x_1 y_1, x_2 y_2, ..., x_D y_D )
#'
#' @param x Composition with strictly positive parts.
#' @param y Second composition, same length as x, strictly positive.
#' @param total Constant kappa the closure sums to.
#'
#' @return List with ``composition``, ``total``, ``D``.
#' @references Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The definitions below were taken instead from Mateu-Figueras, G., Pawlowsky-Glahn, V. and Egozcue, J. J., The normal distribution in some constrained sample spaces, arXiv:0802.2643 (published as SORT 37(1):29-56, 2013), Sect. 4.1, which prints them with equation numbers and attributes them to Aitchison (1982, 1986); that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  Perturbation is the inner sum of the simplex: componentwise multiplication followed by closure.  It plays the role that addition plays in real space, which is why translations of compositional data are perturbations.
#' @export
Compperturb <- function(x, y, total = 1) {
  x <- .t1_vec(x); y <- .t1_vec(y)
  if (length(x) != length(y)) stop("x and y must have the same number of parts")
  if (length(x) == 0L) stop("x must be non-empty")
  if (any(x <= 0) || any(y <= 0)) stop("compositions must be strictly positive")
  p <- x * y; s <- sum(p); k <- as.numeric(total)
  .t1_result(composition = k * p / s, total = k, D = length(x),
             method = "Perturbation on the simplex")
}
