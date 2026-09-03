# SPDX-License-Identifier: AGPL-3.0-or-later

#' Internal helpers shared across the LLM-architecture suite.
#' Not exported; consumed by llm_arch callables only
#' @keywords internal
#' @name llm_arch_helpers
NULL

#' Softmax along the last axis of an array
#'
#' A step of the helpers_llm_arch implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param x A matrix; passed to \code{dim}.
#' @return The value of \code{aperm}.
#' @export
#' @examples
#' x <- c(1.2, 2.4, 3.1, 4.8, 5.3, 6.7, 7.1, 8.9)
#' res <- .softmax_last(x = x)
#' res
.softmax_last <- function(x) {
  # softmax along the last axis of an array
  d <- dim(x)
  nd <- length(d)
  if (is.null(d) || nd == 1L) {
    x <- x - max(x)
    e <- exp(x)
    return(e / sum(e))
  }
  out <- apply(x, seq_len(nd - 1L), function(v) {
    v <- v - max(v)
    e <- exp(v)
    e / sum(e)
  })
  # apply collapses last axis to first; transpose back
  aperm(out, c(seq.int(2L, nd), 1L))
}

`%||%` <- function(a, b) if (is.null(a)) b else a
