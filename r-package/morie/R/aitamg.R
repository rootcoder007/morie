# SPDX-License-Identifier: AGPL-3.0-or-later
#' Amalgamation of a composition into grouped parts.
#'
#' Formula: amalg(x; g)_k = sum_{i : g_i = k} x_i, then closed
#'
#' @param x Composition with strictly positive parts.
#' @param groups Group label 1..k for each part of x, same length as x; every label from 1 to k must be used.
#' @param total Constant kappa the closure sums to.
#'
#' @return List with ``composition``, ``raw``, ``k``, ``total``, ``D``.
#' @references Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read; see EXTERNAL_SOURCES.md.  Amalgamation adds parts together, unlike subcomposition which drops them.  It is the operation that is NOT subcompositionally coherent: log-ratios among the amalgamated groups are not determined by the log-ratios among the original parts, which is why the two operations are kept as separate functions here.  Group labels are 1-based in BOTH language arms.  Implemented in the standard published form.  The log-ratio algebra it rests on was verified against Mateu-Figueras, Pawlowsky-Glahn and Egozcue, arXiv:0802.2643 Sect. 4.1 (fetched and archived), but this particular definition is not printed there and could not be checked against Aitchison's own text.
#' @export
Compamalg <- function(x, groups, total = 1) {
  x <- .t1_vec(x); g <- as.integer(groups)
  if (length(g) != length(x)) stop("groups must have one label per part of x")
  if (length(x) == 0L) stop("x must be non-empty")
  if (any(x <= 0)) stop("compositions must be strictly positive")
  k <- max(g)
  if (min(g) != 1L || !identical(sort(unique(g)), seq_len(k)))
    stop("group labels must be 1..k with every label used")
  raw <- as.numeric(tapply(x, factor(g, levels = seq_len(k)), sum))
  s <- sum(raw); t <- as.numeric(total)
  .t1_result(composition = t * raw / s, raw = raw, k = k, total = t,
             D = length(x), method = "Amalgamation")
}
