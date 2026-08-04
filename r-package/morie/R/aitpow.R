# SPDX-License-Identifier: AGPL-3.0-or-later
#' Powering, the scalar multiplication of the simplex.
#'
#' Formula: a (.) x = C( x_1^a, x_2^a, ..., x_D^a )
#'
#' @param x Composition with strictly positive parts.
#' @param a Real scalar.
#' @param total Constant kappa the closure sums to.
#'
#' @return List with ``composition``, ``alpha``, ``total``, ``D``.
#' @references Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The definitions below were taken instead from Mateu-Figueras, G., Pawlowsky-Glahn, V. and Egozcue, J. J., The normal distribution in some constrained sample spaces, arXiv:0802.2643 (published as SORT 37(1):29-56, 2013), Sect. 4.1, which prints them with equation numbers and attributes them to Aitchison (1982, 1986); that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  Powering is the outer product of the simplex: raise every part to the power a and close.  It plays the role that multiplication by a scalar plays in real space, and satisfies d_a(a(.)x, a(.)x*) = |a| d_a(x, x*).
#' @export
Comppower <- function(x, a, total = 1) {
  x <- .t1_vec(x)
  if (length(x) == 0L) stop("x must be non-empty")
  if (any(x <= 0)) stop("compositions must be strictly positive")
  a <- as.numeric(a); lg <- a * log(x)
  e <- exp(lg - max(lg)); s <- sum(e); k <- as.numeric(total)
  .t1_result(composition = k * e / s, alpha = a, total = k, D = length(x),
             method = "Powering on the simplex")
}
