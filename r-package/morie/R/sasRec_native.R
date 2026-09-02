# SASRec: self-attentive sequential recommendation.
# Sources: Kang, W.-C. & McAuley, J. (2018) "Self-Attentive
# Sequential Recommendation", *Proceedings of the 2018 IEEE
# International Conference on Data Mining (ICDM 2018)*, 197-206,
# doi:10.1109/ICDM.2018.00035, arXiv:1808.09781. The abstract: Markov
# chains assume the next action is predictable from the last few,
# while RNNs allow longer-term semantics; MC-based methods perform
# best in extremely sparse datasets where parsimony is critical,
# RNNs in denser datasets where complexity is affordable; SASRec
# balances these by capturing long-term semantics like an RNN while
# making predictions from relatively few actions like an MC,
# identifying at each step which items are relevant; outperforming
# MC/CNN/RNN baselines on both sparse and dense datasets; being an
# order of magnitude more efficient; and attention-weight
# visualisations showing adaptive handling of datasets of various
# density. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J.,
# Jones, L., Gomez, A. N., Kaiser, L. & Polosukhin, I. (2017)
# "Attention Is All You Need", *NIPS 2017*, 5998-6008,
# arXiv:1706.03762. Hidasi, B., Karatzoglou, A., Baltrunas, L. &
# Tikk, D. (2016) "Session-based Recommendations with Recurrent
# Neural Networks", *ICLR 2016*, arXiv:1511.06939. The RNN baseline;
# implemented in gru4r.

.SASREC_EPS <- 1e-12

#' .sasrec_mat
#'
#' A step of the sasRec_native implementation. Called by \code{attention_span}, \code{predict_next}, \code{self_attention}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x A matrix; passed to \code{as.matrix}.
#' @return Nothing; this branch always raises.
#' @export
.sasrec_mat <- function(x) {
  if (is.matrix(x)) return(x)
  if (is.numeric(x)) return(as.matrix(x))
  if (is.list(x)) {
    n <- length(x)
    d <- length(x[[1]])
    M <- matrix(0, n, d)
    for (i in seq_len(n)) M[i, ] <- as.numeric(x[[i]])
    return(M)
  }
  stop("sasRec: expected a matrix-like input")
}

#' .sasrec_vec
#'
#' A step of the sasRec_native implementation. Called by \code{predict_next}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x A matrix; indexed by row and column.
#' @return A vector, from \code{as.numeric}.
#' @export
.sasrec_vec <- function(x) {
  if (is.matrix(x)) {
    if (nrow(x) == 1L) return(as.numeric(x[1, ]))
    if (ncol(x) == 1L) return(as.numeric(x[, 1]))
  }
  as.numeric(x)
}

#' causal_mask
#'
#' A step of the sasRec_native implementation. Called by \code{self_attention}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param n Coerced to integer by the body, with \code{as.integer}.
#' @return The value of \code{M}, as built in the body.
#' @export
causal_mask <- function(n) {
  m <- as.integer(n)
  if (m < 1L)
    stop("sasRec: the sequence must be non-empty")
  M <- matrix(0, m, m)
  for (i in seq_len(m)) for (j in seq_len(m))
    M[i, j] <- if (j <= i) 1.0 else 0.0
  M
}

#' self_attention
#'
#' A step of the sasRec_native implementation. Called by \code{morie_sasRec}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param E Passed to \code{.sasrec_mat}.
#' @param WQ A matrix; passed to \code{nrow}.
#' @param WK A matrix; passed to \code{ncol}.
#' @param WV A matrix; passed to \code{ncol}.
#' @param mask Optional; may be \code{NULL}. Passed to \code{is.null}.
#' @return A list with \code{output}, \code{weights}, \code{note}.
#' @export
self_attention <- function(E, WQ, WK, WV, mask = NULL) {
  X <- .sasrec_mat(E)
  n <- nrow(X)
  d <- ncol(X)
  M <- if (is.null(mask)) causal_mask(n) else mask
  WQ <- as.matrix(WQ)
  WK <- as.matrix(WK)
  WV <- as.matrix(WV)
  dk <- ncol(WQ)
  if (ncol(WK) != dk || nrow(WQ) != dk)
    stop("sasRec: WQ/WK must share the key dimension")
  if (ncol(WV) != d)
    stop("sasRec: WV must map to the value dimension")
  Q <- X %*% t(WQ)
  K <- X %*% t(WK)
  V <- X %*% t(WV)
  sc <- Q %*% t(K) / sqrt(dk)
  sc[!as.logical(M)] <- -1e30
  mx <- apply(sc, 1, max)
  e <- exp(sc - mx)
  z <- rowSums(e)
  z[z == 0] <- 1
  W <- e / z
  out <- W %*% V
  list(output = out, weights = W,
       note = "the mask is a correctness condition, not an optimisation")
}

#' attention_span
#'
#' A step of the sasRec_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param weights Passed to \code{.sasrec_mat}.
#' @param position Optional; may be \code{NULL}. Coerced to integer by the body, with \code{as.integer}.
#' @return A list with \code{mean_lookback}, \code{mass_on_last}, \code{effective_order}, \code{note}.
#' @export
attention_span <- function(weights, position = NULL) {
  W <- .sasrec_mat(weights)
  i <- if (is.null(position)) nrow(W) else as.integer(position) + 1L
  row <- W[i, ]
  tot <- sum(row[seq_len(i)])
  if (tot <= .SASREC_EPS)
    stop("sasRec: the attention row has no mass")
  span <- sum((i - seq_len(i)) * row[seq_len(i)]) / tot
  list(mean_lookback = span,
       mass_on_last = row[i] / tot,
       effective_order = span + 1,
       note = "a short span IS Markov behaviour; a long one is RNN behaviour, chosen per sequence")
}

#' predict_next
#'
#' A step of the sasRec_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param state Passed to \code{.sasrec_vec}.
#' @param item_embeddings Passed to \code{.sasrec_mat}.
#' @param top_k Coerced to integer by the body, with \code{as.integer}. Defaults to \code{5}.
#' @param exclude Passed to \code{unlist}. Defaults to \code{list()}.
#' @return A list with \code{estimate}, \code{ranking}, \code{n_scored}, \code{method}.
#' @export
predict_next <- function(state, item_embeddings, top_k = 5,
                         exclude = list()) {
  s <- .sasrec_vec(state)
  E <- .sasrec_mat(item_embeddings)
  ex <- as.integer(unlist(exclude))
  sc <- numeric(nrow(E))
  for (i in seq_len(nrow(E))) {
    if (i %in% ex) { sc[i] <- -Inf
    next }
    sc[i] <- sum(s * E[i, ])
  }
  ord <- order(-sc, seq_along(sc))
  ranking <- ord[seq_len(min(as.integer(top_k), length(ord)))]
  list(estimate = ranking, ranking = ranking,
       n_scored = sum(is.finite(sc)),
       method = "self-attentive sequential recommendation; Kang & McAuley (2018)")
}

#' complexity
#'
#' A step of the sasRec_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param n Coerced to integer by the body, with \code{as.integer}.
#' @param d Coerced to integer by the body, with \code{as.integer}.
#' @return A list with \code{attention_ops}, \code{rnn_ops}, \code{attention_sequential_steps}, \code{rnn_sequential_steps}, \code{note}.
#' @export
complexity <- function(n, d) {
  nn <- as.integer(n)
  dd <- as.integer(d)
  if (nn < 1L || dd < 1L)
    stop("sasRec: n and d must be positive")
  list(attention_ops = nn * nn * dd, rnn_ops = nn * dd * dd,
       attention_sequential_steps = 1L, rnn_sequential_steps = nn,
       note = "the parallelism, not the operation count, is where the order-of-magnitude speed-up comes from")
}

#' morie_sasRec
#'
#' A step of the sasRec_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param E Passed to \code{self_attention}.
#' @param WQ Passed to \code{self_attention}.
#' @param WK Passed to \code{self_attention}.
#' @param WV Passed to \code{self_attention}.
#' @param mask Passed to \code{self_attention}.
#' @return The value of \code{self_attention}.
#' @export
morie_sasRec <- function(E, WQ, WK, WV, mask = NULL) {
  self_attention(E, WQ, WK, WV, mask = mask)
}

sasrec <- self_attention
selfattentivesequential <- self_attention

#' .sasRec_cheatsheet
#'
#' A step of the sasRec_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @return A character value.
#' @export
.sasRec_cheatsheet <- function() {
  paste("sasRec: Markov chains win where data are SPARSE (parsimony",
        "is critical), RNNs where they are DENSE (complexity is",
        "affordable) -- and the choice is normally made once for a",
        "whole dataset. Self-attention picks per sequence: it can",
        "reach far back like an RNN while predicting from FEW",
        "actions like an MC, and the attention weights show it",
        "adapting to density. Causal masking is a CORRECTNESS",
        "condition -- attending forward leaks the target. O(n^2 d)",
        "but fully parallel against an RNN's inherently sequential",
        "O(n d^2).")
}
