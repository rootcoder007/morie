# SPDX-License-Identifier: AGPL-3.0-or-later
#' Additive log-ratio transform against a chosen reference part.
#'
#' \code{ref} is a one-based part index; the remaining parts keep their
#' original order.
#'
#' Formula: alr(x)_i = log( x_i / x_ref ), i != ref
#'
#' @param x Strictly positive vector of parts.
#' @param ref One-based index of the reference part (default: the last).
#' @return List with \code{alr}, \code{ref}, \code{kept}, \code{D}.
#' @references Aitchison (1986), The Statistical Analysis of Compositional
#'   Data, Chapter 4. Verified against the reference implementation in the
#'   CRAN package compositions 2.0-9, whose alr defaults its ivar to the
#'   last column and returns log(x_i) - log(x_ivar).
#' @export
Alr <- function(x, ref = NULL) {
  x <- .t1_vec(x)
  if (any(x <= 0)) stop("compositions must be strictly positive")
  D <- length(x)
  r <- if (is.null(ref)) D else as.integer(ref)
  if (r < 1L || r > D) stop("ref must be a one-based part index in 1..D")
  keep <- seq_len(D)[-r]
  .t1_result(alr = log(x[keep]) - log(x[r]), ref = r, kept = keep, D = D,
             method = "Additive log-ratio transform")
}
