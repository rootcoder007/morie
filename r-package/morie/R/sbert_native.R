# Sentence-BERT: sentence embeddings that can actually be compared.
# Sources: Reimers, N. & Gurevych, I. (2019) "Sentence-BERT: Sentence
# Embeddings using Siamese BERT-Networks", *Proceedings of the 2019
# Conference on Empirical Methods in Natural Language Processing and
# the 9th International Joint Conference on Natural Language
# Processing (EMNLP-IJCNLP)*, 3980-3990, doi:10.18653/v1/D19-1410,
# arXiv:1908.10084. Sec. 2 (BERT's sentence-pair regression setup
# with [SEP] and its cost; the absence of independent sentence
# embeddings; the averaging and [CLS] workarounds and the observation
# that they were unevaluated) and the architecture figures giving the
# softmax objective over (u, v, |u-v|) and the cosine-similarity
# objective. Conneau, A., Kiela, D., Schwenk, H., Barrault, L. &
# Bordes, A. (2017) "Supervised Learning of Universal Sentence
# Representations from Natural Language Inference Data", *EMNLP
# 2017*, 670-680, arXiv:1705.02364. InferSent: the siamese BiLSTM
# with max pooling trained on NLI that this follows. Devlin, J.,
# Chang, M.-W., Lee, K. & Toutanova, K. (2019) "BERT: Pre-training of
# Deep Bidirectional Transformers for Language Understanding",
# *NAACL-HLT 2019*, 4171-4186, arXiv:1810.04805.

.SBERT_EPS <- 1e-12
.SBERT_POOLING <- c("mean", "cls", "max")

#' .sbert_mat
#'
#' A step of the sbert_native implementation. Called by \code{pool}, \code{rank_by_similarity}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x A matrix; passed to \code{as.matrix}.
#' @return Nothing; this branch always raises.
#' @export
#' @examples
#' x <- c(1.2, 2.4, 3.1, 4.8, 5.3, 6.7, 7.1, 8.9)
#' res <- .sbert_mat(x = x)
#' res
.sbert_mat <- function(x) {
  if (is.matrix(x)) return(x)
  if (is.numeric(x)) return(as.matrix(x))
  if (is.list(x)) {
    n <- length(x)
    d <- length(x[[1]])
    M <- matrix(0, n, d)
    for (i in seq_len(n)) M[i, ] <- as.numeric(x[[i]])
    return(M)
  }
  stop("sbert: expected a matrix-like input")
}

#' .sbert_vec
#'
#' A step of the sbert_native implementation. Called by \code{classification_features},
#' \code{cosine_similarity}, \code{sts_score}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x A matrix; indexed by row and column.
#' @return A vector, from \code{as.numeric}.
#' @export
#' @examples
#' x <- c(1.2, 2.4, 3.1, 4.8, 5.3, 6.7, 7.1, 8.9)
#' res <- .sbert_vec(x = x)
#' res
.sbert_vec <- function(x) {
  if (is.matrix(x)) {
    if (nrow(x) == 1L) return(as.numeric(x[1, ]))
    if (ncol(x) == 1L) return(as.numeric(x[, 1]))
  }
  as.numeric(x)
}

#' pool
#'
#' A step of the sbert_native implementation. Called by \code{morie_geron_lenet5}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param token_vectors Passed to \code{.sbert_mat}.
#' @param mode One of \code{"cls"}, \code{"max"}. Defaults to \code{"mean"}.
#' @param mask Optional; may be \code{NULL}. Coerced to logical by the body, with \code{as.logical}.
#' @return A numeric value.
#' @export
pool <- function(token_vectors, mode = "mean", mask = NULL) {
  if (!(mode %in% .SBERT_POOLING))
    stop("sbert: pooling must be one of mean, cls, max, got ", format(mode))
  T <- .sbert_mat(token_vectors)
  if (nrow(T) == 0L)
    stop("sbert: no token vectors given")
  d <- ncol(T)
  m <- if (is.null(mask)) rep(TRUE, nrow(T)) else as.logical(mask)
  if (length(m) != nrow(T))
    stop("sbert: ", length(m), " mask entries for ", nrow(T), " tokens")
  keep <- which(m)
  if (length(keep) == 0L)
    stop("sbert: the mask excludes every token")
  if (mode == "cls") return(as.numeric(T[keep[1L], ]))
  if (mode == "max") {
    out <- numeric(d)
    for (j in seq_len(d)) out[j] <- max(T[keep, j])
    return(out)
  }
  out <- numeric(d)
  for (j in seq_len(d)) out[j] <- sum(T[keep, j])
  out / length(keep)
}

#' cosine_similarity
#'
#' A step of the sbert_native implementation. Called by \code{rank_by_similarity}, \code{sts_score}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param u Passed to \code{.sbert_vec}.
#' @param v Passed to \code{.sbert_vec}.
#' @return A numeric value.
#' @export
cosine_similarity <- function(u, v) {
  a <- as.numeric(.sbert_vec(u))
  b <- as.numeric(.sbert_vec(v))
  if (length(a) != length(b))
    stop("sbert: vectors differ in length (", length(a), ", ", length(b), ")")
  na <- sqrt(sum(a^2))
  nb <- sqrt(sum(b^2))
  if (na <= .SBERT_EPS || nb <= .SBERT_EPS)
    stop("sbert: cosine similarity is undefined for a zero vector")
  sum(a * b) / (na * nb)
}

#' classification_features
#'
#' A step of the sbert_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param u Passed to \code{.sbert_vec}.
#' @param v Passed to \code{.sbert_vec}.
#' @return A list with \code{features}, \code{u}, \code{v}, \code{abs_diff}, \code{dim},
#' \code{note}.
#' @export
classification_features <- function(u, v) {
  a <- as.numeric(.sbert_vec(u))
  b <- as.numeric(.sbert_vec(v))
  if (length(a) != length(b))
    stop("sbert: vectors differ in length (", length(a), ", ", length(b), ")")
  diff <- abs(a - b)
  list(features = c(a, b, diff), u = a, v = b,
       abs_diff = diff, dim = 3 * length(a),
       note = "|u - v| is the term neither u nor v supplies")
}

#' pair_cost
#'
#' A step of the sbert_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param n Coerced to integer by the body, with \code{as.integer}.
#' @param mode One of \code{"bi-encoder"}, \code{"cross-encoder"}. Defaults to
#' \code{"cross-encoder"}.
#' @return A list with \code{forward_passes}, \code{cross_encoder}, \code{bi_encoder},
#' \code{speedup}, \code{n}, \code{note}.
#' @export
pair_cost <- function(n, mode = "cross-encoder") {
  N <- as.integer(n)
  if (N < 2L)
    stop("sbert: need at least 2 sentences")
  if (!(mode %in% c("cross-encoder", "bi-encoder")))
    stop("sbert: mode must be cross-encoder or bi-encoder, got ", format(mode))
  cross <- (N * (N - 1L)) %/% 2L
  list(forward_passes = if (mode == "cross-encoder") cross else N,
       cross_encoder = cross, bi_encoder = N,
       speedup = cross / N, n = N,
       note = "the bi-encoder also does O(n^2) dot products, but those are arithmetic, not network passes")
}

#' rank_by_similarity
#'
#' A step of the sbert_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param query Passed to \code{cosine_similarity}.
#' @param corpus_embeddings Passed to \code{.sbert_mat}.
#' @param top_k Coerced to integer by the body, with \code{as.integer}. Defaults to \code{5}.
#' @return A list with \code{ranking}, \code{n_corpus}, \code{forward_passes}, \code{note}.
#' @export
rank_by_similarity <- function(query, corpus_embeddings, top_k = 5) {
  E <- .sbert_mat(corpus_embeddings)
  if (nrow(E) == 0L)
    stop("sbert: the corpus is empty")
  scores <- lapply(seq_len(nrow(E)), function(i) {
    c(i, cosine_similarity(query, E[i, , drop = FALSE]))
  })
  ord <- order(-sapply(scores, function(s) s[2]))
  top <- ord[seq_len(min(as.integer(top_k), length(ord)))]
  list(ranking = scores[top], n_corpus = nrow(E),
       forward_passes = 0,
       note = "no network passes at query time -- the corpus was embedded once")
}

#' sts_score
#'
#' A step of the sbert_native implementation. Called by \code{morie_sbert}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param pairs A vector; its length is taken and its elements indexed.
#' @param embed Accepted by the signature and not used anywhere in the body.
#' @return A list with \code{estimate}, \code{scores}, \code{embed_calls},
#' \code{n_pairs}, \code{cross_encoder_calls}, \code{method}.
#' @export
sts_score <- function(pairs, embed) {
  cache <- list()
  out <- numeric(length(pairs))
  calls <- 0L
  for (i in seq_along(pairs)) {
    a <- pairs[[i]][1]
    b <- pairs[[i]][2]
    for (s in c(a, b)) {
      key <- as.character(s)
      if (is.null(cache[[key]])) {
        cache[[key]] <- as.numeric(.sbert_vec(embed(s)))
        calls <- calls + 1L
      }
    }
    out[i] <- cosine_similarity(cache[[as.character(a)]],
                                cache[[as.character(b)]])
  }
  list(estimate = out, scores = out, embed_calls = calls,
       n_pairs = length(pairs),
       cross_encoder_calls = length(pairs),
       method = "siamese bi-encoder scored by cosine; Reimers & Gurevych (2019)")
}

#' morie_sbert
#'
#' A step of the sbert_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param pairs Passed to \code{sts_score}.
#' @param embed Passed to \code{sts_score}.
#' @return The value of \code{sts_score}.
#' @export
morie_sbert <- function(pairs, embed) {
  sts_score(pairs, embed)
}

sbert <- sts_score
sentencebert <- sts_score

#' .sbert_cheatsheet
#'
#' A step of the sbert_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @return A character value.
#' @export
#' @examples
#' res <- .sbert_cheatsheet()
#' res
.sbert_cheatsheet <- function() {
  paste("sbert: BERT scores a PAIR, so comparing n sentences needs",
        "C(n,2) forward passes -- 10k sentences is ~50M. A SIAMESE",
        "network embeds each sentence ONCE with shared weights, so",
        "it is n passes plus dot products. Classification",
        "objective: softmax over (u, v, |u-v|) -- the difference",
        "term is what locates the disagreement. Regression",
        "objective: cosine directly, and only that one trains the",
        "cosine geometry. Pooling (mean/CLS/max) is a real choice,",
        "all three ablated.")
}
