# SPDX-License-Identifier: AGPL-3.0-or-later
#' Closure of a composition to a constant sum.
#'
#' Formula: C(x)_i = kappa * x_i / sum_j x_j
#'
#' @param x Composition with strictly positive parts.
#' @param total Constant kappa the closure sums to.
#'
#' @return List with ``composition``, ``total``, ``sum_raw``, ``D``.
#' @references Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The definitions below were taken instead from Mateu-Figueras, G., Pawlowsky-Glahn, V. and Egozcue, J. J., The normal distribution in some constrained sample spaces, arXiv:0802.2643 (published as SORT 37(1):29-56, 2013), Sect. 4.1, which prints them with equation numbers and attributes them to Aitchison (1982, 1986); that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  The closure operator C normalises any positive vector to the constant sum kappa that defines the simplex S^D.
#' @export
Compclose <- function(x, total = 1) {
  x <- .t1_vec(x)
  if (length(x) == 0L) stop("x must be non-empty")
  if (any(x <= 0)) stop("compositions must be strictly positive")
  s <- sum(x); k <- as.numeric(total)
  .t1_result(composition = k * x / s, total = k, sum_raw = s, D = length(x),
             method = "Closure to a constant sum")
}
