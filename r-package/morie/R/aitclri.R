# SPDX-License-Identifier: AGPL-3.0-or-later
#' Map clr coordinates back to a closed composition.
#'
#' Formula: clr^-1(z) = C( exp(z_1), ..., exp(z_D) )
#'
#' @param z clr coordinates.
#' @param total Constant the returned composition sums to.
#' @return List with \code{composition}, \code{shift}, \code{total}, \code{D}.
#' @references Aitchison (1986), The Statistical Analysis of Compositional
#'   Data, Chapter 4. Verified against the reference implementation in the
#'   CRAN package compositions 2.0-9, whose clrInv is acomp(exp(z)), i.e.
#'   the closure of the exponentiated argument.
#' @export
Clrinv <- function(z, total = 1) {
  z <- .t1_vec(z)
  D <- length(z)
  shift <- sum(z) / D
  e <- exp(z - shift)
  s <- sum(e)
  k <- as.numeric(total)
  .t1_result(composition = k * e / s, shift = shift, total = k, D = D,
             method = "Inverse centred log-ratio transform")
}
