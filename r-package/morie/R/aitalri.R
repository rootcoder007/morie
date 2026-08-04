# SPDX-License-Identifier: AGPL-3.0-or-later
#' Map alr coordinates back to a closed composition.
#'
#' \code{ref} is a one-based index into the reconstructed composition of
#' length D = length(y) + 1.
#'
#' Formula: alr^-1(y) = C( exp(y_1), ..., 1 at position ref, ... )
#'
#' @param y alr coordinates, length D-1.
#' @param ref One-based position the reference part is reinstated at.
#' @param total Constant the returned composition sums to.
#' @return List with \code{composition}, \code{ref}, \code{total}, \code{D}.
#' @references Aitchison (1986), The Statistical Analysis of Compositional
#'   Data, Chapter 4. Verified against the reference implementation in the
#'   CRAN package compositions 2.0-9, whose alrInv is clo(exp(cbind(z, 0)))
#'   -- a zero column appended, then closed.
#' @export
Alrinv <- function(y, ref = NULL, total = 1) {
  y <- .t1_vec(y)
  D <- length(y) + 1L
  r <- if (is.null(ref)) D else as.integer(ref)
  if (r < 1L || r > D) stop("ref must be a one-based part index in 1..D")
  full <- append(y, 0, after = r - 1L)
  e <- exp(full)
  s <- sum(e)
  k <- as.numeric(total)
  .t1_result(composition = k * e / s, ref = r, total = k, D = D,
             method = "Inverse additive log-ratio transform")
}
