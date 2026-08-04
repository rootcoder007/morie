# SPDX-License-Identifier: AGPL-3.0-or-later
#' Subcomposition formed from a selected set of parts.
#'
#' Formula: sub(x; S) = C( x_i : i in S )
#'
#' @param x Composition with strictly positive parts.
#' @param parts 1-based indices of the parts to keep; at least two, no repeats.
#' @param total Constant kappa the closure sums to.
#'
#' @return List with ``composition``, ``parts``, ``total``, ``D``, ``D_full``.
#' @references Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read; see EXTERNAL_SOURCES.md.  A subcomposition is the closure of a selected subset of parts.  Log-ratios between retained parts are unchanged by the operation, which is the property that makes subcompositional coherence a requirement on compositional methods.  Indices are 1-based in BOTH language arms.  Implemented in the standard published form.  The log-ratio algebra it rests on was verified against Mateu-Figueras, Pawlowsky-Glahn and Egozcue, arXiv:0802.2643 Sect. 4.1 (fetched and archived), but this particular definition is not printed there and could not be checked against Aitchison's own text.
#' @export
Compsubcomp <- function(x, parts, total = 1) {
  x <- .t1_vec(x); D0 <- length(x)
  if (any(x <= 0)) stop("compositions must be strictly positive")
  idx <- as.integer(parts)
  if (length(idx) < 2L) stop("a subcomposition needs at least two parts")
  if (anyDuplicated(idx)) stop("parts must not repeat")
  if (any(idx < 1L | idx > D0)) stop("parts must be 1-based indices into x")
  sub <- x[idx]; s <- sum(sub); k <- as.numeric(total)
  .t1_result(composition = k * sub / s, parts = idx, total = k,
             D = length(idx), D_full = D0, method = "Subcomposition")
}
