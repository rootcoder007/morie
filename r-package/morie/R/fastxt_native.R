# morie.fn -- function file (rootcoder007/morie)
# FastText -- word vectors enriched with subword information.
#
# References
# Bojanowski, P., Grave, E., Joulin, A. & Mikolov, T. (2017) "Enriching
# word vectors with subword information", Transactions of the Association
# for Computational Linguistics 5, 135-146, arXiv:1607.04606. Sec. 3.2
# for the subword model and the *where* example, Sec. 3.1 for the
# objective.
# Mikolov, T., Sutskever, I., Chen, K., Corrado, G. & Dean, J. (2013)
# "Distributed representations of words and phrases and their
# compositionality", NIPS 26, 3111-3119 -- the skipgram with negative
# sampling this extends.

#' .subwords
#'
#' Part of the fastxt_native implementation; see the file header for the
#' source it follows.
#'
#' @param word See Usage.
#' @param n_min Defaults to \code{3L}.
#' @param n_max Defaults to \code{6L}.
#' @param boundary Defaults to \code{TRUE}.
#' @param whole_word Defaults to \code{TRUE}.
#' @return The value of \code{grams}, as built in the body.
#' @export
.subwords <- function(word, n_min = 3L, n_max = 6L, boundary = TRUE,
                      whole_word = TRUE) {
  lo <- as.integer(n_min); hi <- as.integer(n_max)
  if (lo < 1L) stop(sprintf("subwords: n_min must be at least 1, got %s", format(n_min)))
  if (hi < lo) stop(sprintf("subwords: n_max (%s) is below n_min (%s)", format(n_max), format(n_min)))
  w <- as.character(word)
  padded <- if (isTRUE(boundary)) paste0("<", w, ">") else w
  grams <- character(0)
  seen <- character(0)
  for (n in lo:hi) {
    for (i in 1:(nchar(padded) - n + 1L)) {
      g <- substr(padded, i, i + n - 1L)
      if (!(g %in% seen)) {
        seen <- c(seen, g)
        grams <- c(grams, g)
      }
    }
  }
  if (isTRUE(whole_word)) {
    special <- if (isTRUE(boundary)) paste0("<", w, ">") else w
    if (!(special %in% seen)) grams <- c(grams, special)
  }
  grams
}

#' .fnv1a
#'
#' Part of the fastxt_native implementation; see the file header for the
#' source it follows.
#'
#' @param s See Usage.
#' @return The value of \code{h}, as built in the body.
#' @export
.fnv1a <- function(s) {
  h <- 2166136261
  bytes <- as.integer(charToRaw(s))
  for (ch in bytes) {
    h <- bitwXor(h, ch)
    h <- bitwAnd(h * 16777619, 0xFFFFFFFF)
  }
  h
}

#' .gram_slot
#'
#' Part of the fastxt_native implementation; see the file header for the
#' source it follows.
#'
#' @param g See Usage.
#' @param gram_index See Usage.
#' @param hash_buckets See Usage.
#' @return The value of \code{gi}, as built in the body.
#' @export
.gram_slot <- function(g, gram_index, hash_buckets) {
  if (!is.null(hash_buckets)) {
    return(.fnv1a(g) %% as.integer(hash_buckets))
  }
  gi <- gram_index[[g]]
  if (is.null(gi)) return(NULL)
  gi
}

#' .word_vector
#'
#' Part of the fastxt_native implementation; see the file header for the
#' source it follows.
#'
#' @param word See Usage.
#' @param Z See Usage.
#' @param gram_index See Usage.
#' @param n_min Defaults to \code{3L}.
#' @param n_max Defaults to \code{6L}.
#' @param boundary Defaults to \code{TRUE}.
#' @param whole_word Defaults to \code{TRUE}.
#' @param hash_buckets Defaults to \code{NULL}.
#' @return A list with \code{v}, \code{hit}.
#' @export
.word_vector <- function(word, Z, gram_index, n_min = 3L, n_max = 6L,
                         boundary = TRUE, whole_word = TRUE,
                         hash_buckets = NULL) {
  dim_ <- ncol(Z)
  if (is.null(dim_)) dim_ <- 0L
  v <- rep(0, dim_)
  hit <- 0L
  for (g in .subwords(word, n_min, n_max, boundary, whole_word)) {
    idx <- .gram_slot(g, gram_index, hash_buckets)
    if (is.null(idx)) next
    hit <- hit + 1L
    for (t in seq_len(dim_)) v[t] <- v[t] + Z[idx, t]
  }
  list(v = v, hit = hit)
}

#' .as_docs
#'
#' Part of the fastxt_native implementation; see the file header for the
#' source it follows.
#'
#' @param corpus See Usage.
#' @return The value of \code{docs}, as built in the body.
#' @export
.as_docs <- function(corpus) {
  if (is.null(corpus)) stop("fasttext: corpus must not be None")
  docs <- list()
  for (item in corpus) {
    if (is.character(item) && length(item) == 1L) {
      docs[[length(docs) + 1L]] <- strsplit(item, "\\s+")[[1]]
    } else {
      docs[[length(docs) + 1L]] <- as.character(unlist(item))
    }
  }
  if (length(docs) == 0L) stop("fasttext: the corpus is empty")
  docs
}

#' fasttext
#'
#' Part of the fastxt_native implementation; see the file header for the
#' source it follows.
#'
#' @param corpus See Usage.
#' @param dim Defaults to \code{50L}.
#' @param n_min Defaults to \code{3L}.
#' @param n_max Defaults to \code{6L}.
#' @param window Defaults to \code{5L}.
#' @param epochs Defaults to \code{5L}.
#' @param lr Defaults to \code{0.05}.
#' @param negative Defaults to \code{5L}.
#' @param min_count Defaults to \code{1L}.
#' @param boundary Defaults to \code{TRUE}.
#' @param whole_word Defaults to \code{TRUE}.
#' @param hash_buckets Defaults to \code{NULL}.
#' @param seed Defaults to \code{0L}.
#' @return A list with \code{estimate}, \code{vectors}, \code{vocab}, \code{index}, \code{ngrams}, \code{ngram_index}, \code{Z}, \code{context}, \code{loss_history}, \code{final_loss}, \code{oov}, \code{n_vocab}, \code{n_ngrams}, \code{dim}, \code{n_min}, \code{n_max}, \code{hash_buckets}, \code{method}.
#' @export
fasttext <- function(corpus, dim = 50L, n_min = 3L, n_max = 6L,
                     window = 5L, epochs = 5L, lr = 0.05, negative = 5L,
                     min_count = 1L, boundary = TRUE, whole_word = TRUE,
                     hash_buckets = NULL, seed = 0L) {
  docs <- .as_docs(corpus)
  d <- as.integer(dim)
  if (d < 1L) stop(sprintf("fasttext: dim must be at least 1, got %s", format(dim)))
  counts <- list()
  for (doc in docs) for (t in doc) {
    counts[[t]] <- if (is.null(counts[[t]])) 1L else counts[[t]] + 1L
  }
  vocab <- sort(names(counts)[vapply(counts, function(c) c >= as.integer(min_count), logical(1))])
  if (length(vocab) < 2L) {
    stop(sprintf("fasttext: %d word(s) above min_count=%s; skipgram needs a context to predict",
                 length(vocab), format(min_count)))
  }
  windex <- stats::setNames(seq_along(vocab) - 1L, vocab)

  grams <- character(0)
  gram_index <- list()
  for (wd in vocab) {
    for (g in .subwords(wd, n_min, n_max, boundary, whole_word)) {
      if (is.null(gram_index[[g]])) {
        gram_index[[g]] <- length(grams)
        grams <- c(grams, g)
      }
    }
  }
  n_slots <- if (!is.null(hash_buckets)) as.integer(hash_buckets) else length(grams)

  rng <- .ghc_rng(as.integer(seed))
  sc <- 0.5 / d
  Z <- matrix(.ghc_unif(rng, n_slots * d) - 0.5, nrow = n_slots, ncol = d) * sc
  Vc <- matrix(0, nrow = length(vocab), ncol = d)

  freqs <- vapply(vocab, function(t) counts[[t]] ^ 0.75, numeric(1))
  tot <- sum(freqs)
  cum <- cumsum(freqs / tot)

  draw_negative <- function() {
    u <- .ghc_unif(rng, 1L)
    lo <- 1L; hi <- length(cum)
    while (lo < hi) {
      mid <- (lo + hi) %/% 2L
      if (u > cum[mid]) lo <- mid + 1L else hi <- mid
    }
    lo - 1L
  }

  eta <- as.numeric(lr)
  losses <- numeric(0)
  for (ep in seq_len(as.integer(epochs))) {
    total <- 0; n_upd <- 0L
    for (doc in docs) {
      ids <- as.integer(vapply(doc, function(t) {
        w <- windex[[t]]
        if (is.null(w)) NA_integer_ else w
      }, integer(1)))
      ids <- ids[!is.na(ids)]
      for (pos in seq_along(ids)) {
        wd <- vocab[ids[pos] + 1L]
        slots <- vapply(.subwords(wd, n_min, n_max, boundary, whole_word),
                        function(g) {
                          r <- .gram_slot(g, gram_index, hash_buckets)
                          if (is.null(r)) NA_integer_ else r
                        }, integer(1))
        slots <- slots[!is.na(slots)]
        if (length(slots) == 0L) next
        u <- colSums(Z[slots + 1L, , drop = FALSE])
        lo_i <- max(1L, pos - as.integer(window))
        hi_i <- min(length(ids), pos + as.integer(window))
        for (other in lo_i:hi_i) {
          if (other == pos) next
          ctx_idx <- c(ids[other], vapply(seq_len(as.integer(negative)),
                                          function(i) draw_negative(),
                                          integer(1)))
          labels <- c(1.0, rep(0.0, as.integer(negative)))
          grad_u <- rep(0, d)
          for (k in seq_along(ctx_idx)) {
            ci <- ctx_idx[k] + 1L
            label <- labels[k]
            dot <- sum(u * Vc[ci, ])
            dot_c <- max(-30, min(30, dot))
            p <- 1 / (1 + exp(-dot_c))
            g <- p - label
            total <- total - (if (label > 0.5) log(p + 1e-12) else log(1 - p + 1e-12))
            n_upd <- n_upd + 1L
            grad_u <- grad_u + g * Vc[ci, ]
            Vc[ci, ] <- Vc[ci, ] - eta * g * u
          }
          for (s in slots) Z[s + 1L, ] <- Z[s + 1L, ] - eta * grad_u
        }
      }
    }
    losses <- c(losses, if (n_upd > 0L) total / n_upd else NaN)
  }

  vecs <- matrix(0, nrow = length(vocab), ncol = d)
  for (i in seq_along(vocab)) {
    r <- .word_vector(vocab[i], Z, gram_index, n_min, n_max, boundary,
                      whole_word, hash_buckets)
    vecs[i, ] <- r$v
  }

  oov <- function(word) {
    .word_vector(word, Z, gram_index, n_min, n_max, boundary,
                 whole_word, hash_buckets)$v
  }

  list(
    estimate = vecs, vectors = vecs, vocab = vocab, index = windex,
    ngrams = grams, ngram_index = gram_index, Z = Z, context = Vc,
    loss_history = losses, final_loss = if (length(losses)) losses[length(losses)] else NaN,
    oov = oov, n_vocab = length(vocab), n_ngrams = length(grams), dim = d,
    n_min = as.integer(n_min), n_max = as.integer(n_max),
    hash_buckets = hash_buckets,
    method = "fastText subword skipgram with negative sampling, Bojanowski, Grave, Joulin & Mikolov (2017) Sec. 3.2"
  )
}

#' morie_fastxt
#'
#' Part of the fastxt_native implementation; see the file header for the
#' source it follows.
#'
#' @param corpus See Usage.
#' @param dim Defaults to \code{50L}.
#' @param n_min Defaults to \code{3L}.
#' @param n_max Defaults to \code{6L}.
#' @param window Defaults to \code{5L}.
#' @param epochs Defaults to \code{5L}.
#' @param lr Defaults to \code{0.05}.
#' @param negative Defaults to \code{5L}.
#' @param min_count Defaults to \code{1L}.
#' @param boundary Defaults to \code{TRUE}.
#' @param whole_word Defaults to \code{TRUE}.
#' @param hash_buckets Defaults to \code{NULL}.
#' @param seed Defaults to \code{0L}.
#' @return The value of \code{fasttext}.
#' @export
morie_fastxt <- function(corpus, dim = 50L, n_min = 3L, n_max = 6L,
                         window = 5L, epochs = 5L, lr = 0.05,
                         negative = 5L, min_count = 1L, boundary = TRUE,
                         whole_word = TRUE, hash_buckets = NULL,
                         seed = 0L) {
  fasttext(corpus = corpus, dim = dim, n_min = n_min, n_max = n_max,
           window = window, epochs = epochs, lr = lr, negative = negative,
           min_count = min_count, boundary = boundary,
           whole_word = whole_word, hash_buckets = hash_buckets,
           seed = seed)
}
