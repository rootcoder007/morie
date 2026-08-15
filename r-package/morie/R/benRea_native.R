# Named-entity recognition with BIO tagging.
# Sources: Ramshaw, L. A. & Marcus, M. P. (1995) "Text Chunking using
# Transformation-Based Learning", *Proceedings of the Third Workshop on
# Very Large Corpora*, 82-94, arXiv:cmp-lg/9505040, for the IOB/BIO
# tagging scheme. Tjong Kim Sang, E. F. & De Meulder, F. (2003)
# "Introduction to the CoNLL-2003 Shared Task: Language-Independent
# Named Entity Recognition", *Proceedings of CoNLL-2003*, 142-147,
# for the span-level evaluation. Lample, G., Ballesteros, M.,
# Subramanian, S., Kawakami, K. & Dyer, C. (2016) "Neural Architectures
# for Named Entity Recognition", *Proceedings of NAACL-HLT 2016*,
# 260-270, doi:10.18653/v1/N16-1030, arXiv:1603.01360, for the BiLSTM-CRF
# whose constrained decoding this implements. Viterbi, A. J. (1967)
# "Error bounds for convolutional codes and an asymptotically optimum
# decoding algorithm", *IEEE Transactions on Information Theory* 13(2),
# 260-269, doi:10.1109/TIT.1967.1054010. Devlin, J., Chang, M.-W.,
# Lee, K. & Toutanova, K. (2019) "BERT: Pre-training of Deep
# Bidirectional Transformers for Language Understanding",
# *Proceedings of NAACL-HLT 2019*, 4171-4186,
# doi:10.18653/v1/N19-1423.

.benRea_NEG <- -Inf

bio_labels <- function(types) {
  ts <- as.list(types)
  if (length(ts) == 0L) stop("benRea: no entity types given")
  if (any(duplicated(ts))) stop("benRea: duplicate entity types")
  out <- list("O")
  for (t in ts) {
    out[[length(out) + 1L]] <- paste0("B-", t)
    out[[length(out) + 1L]] <- paste0("I-", t)
  }
  out
}

.parts <- function(label) {
  if (label == "O") return(list("O", NA))
  strsplit(label, "-", fixed = TRUE)[[1]]
}

valid_transitions <- function(labels) {
  n <- length(labels)
  T <- matrix(TRUE, n, n)
  for (a in seq_len(n)) {
    pa <- .parts(labels[[a]])[[1]]; ta <- .parts(labels[[a]])[[2]]
    for (b in seq_len(n)) {
      pb <- .parts(labels[[b]])[[1]]; tb <- .parts(labels[[b]])[[2]]
      if (pb == "I")
        T[a, b] <- (pa %in% c("B", "I")) && ta == tb
    }
  }
  T
}

start_allowed <- function(labels) {
  sapply(labels, function(v) .parts(v)[[1]] != "I")
}

is_valid_bio <- function(path, labels = NULL) {
  prev <- "O"; prev_t <- NA
  for (lab in path) {
    p <- .parts(lab)[[1]]; t <- .parts(lab)[[2]]
    if (p == "I" && !(prev %in% c("B", "I") && prev_t == t))
      return(FALSE)
    prev <- p; prev_t <- t
  }
  TRUE
}

greedy_decode <- function(emissions, labels) {
  sapply(seq_along(emissions), function(t) {
    row <- emissions[[t]]
    labels[[which.max(row)]]
  })
}

viterbi_decode <- function(emissions, labels, transitions = NULL,
                           transition_scores = NULL) {
  L <- length(emissions)
  n <- length(labels)
  if (L == 0L) stop("benRea: empty emission sequence")
  if (any(sapply(emissions, length) != n))
    stop("benRea: emissions must have one score per label")
  T_mat <- if (is.null(transitions)) valid_transitions(labels) else transitions
  S <- if (is.null(transition_scores)) matrix(0.0, n, n) else transition_scores
  ok0 <- start_allowed(labels)
  dp <- matrix(.benRea_NEG, L, n)
  bk <- matrix(-1L, L, n)
  for (j in seq_len(n)) {
    if (ok0[j]) dp[1L, j] <- emissions[[1L]][j]
  }
  for (t in 2:L) {
    for (j in seq_len(n)) {
      best <- .benRea_NEG; arg <- -1L
      for (i in seq_len(n)) {
        if (!T_mat[i, j] || dp[t - 1L, i] == .benRea_NEG) next
        v <- dp[t - 1L, i] + S[i, j]
        if (v > best) { best <- v; arg <- i }
      }
      if (arg >= 1L) {
        dp[t, j] <- best + emissions[[t]][j]
        bk[t, j] <- arg
      }
    }
  }
  end <- which.max(dp[L, ])
  if (dp[L, end] == .benRea_NEG) stop("benRea: no valid path exists")
  path_idx <- end
  for (t in L:2) {
    path_idx <- c(bk[t, path_idx[1L]], path_idx)
  }
  rev_idx <- rev(path_idx)
  list(path = labels[rev_idx], score = dp[L, end])
}

extract_spans <- function(path) {
  spans <- list()
  cur_t <- NULL; cur_s <- NULL
  for (i in seq_along(path)) {
    lab <- path[[i]]
    p <- .parts(lab)[[1]]; ty <- .parts(lab)[[2]]
    if (p == "B") {
      if (!is.null(cur_t))
        spans[[length(spans) + 1L]] <- list(type = cur_t, start = cur_s, end = i - 1L)
      cur_t <- ty; cur_s <- i
    } else if (p == "I") {
      if (!identical(cur_t, ty)) {
        if (!is.null(cur_t))
          spans[[length(spans) + 1L]] <- list(type = cur_t, start = cur_s, end = i - 1L)
        cur_t <- ty; cur_s <- i
      }
    } else {
      if (!is.null(cur_t))
        spans[[length(spans) + 1L]] <- list(type = cur_t, start = cur_s, end = i - 1L)
      cur_t <- NULL; cur_s <- NULL
    }
  }
  if (!is.null(cur_t))
    spans[[length(spans) + 1L]] <- list(type = cur_t, start = cur_s, end = length(path))
  spans
}

span_f1 <- function(pred, gold) {
  p <- extract_spans(pred); g <- extract_spans(gold)
  sp <- function(spans) {
    sapply(spans, function(s) paste0(s$type, "|", s$start, "|", s$end))
  }
  pset <- sp(p); gset <- sp(g)
  tp <- length(intersect(pset, gset))
  prec <- if (length(pset) > 0L) tp / length(pset) else 0.0
  rec <- if (length(gset) > 0L) tp / length(gset) else 0.0
  f1 <- if ((prec + rec) > 0) 2 * prec * rec / (prec + rec) else 0.0
  list(precision = prec, recall = rec, f1 = f1,
       true_positives = tp, n_pred = length(pset), n_gold = length(gset))
}

ner_decode <- function(emissions, types, decoder = "viterbi",
                       transition_scores = NULL, gold = NULL) {
  if (!(decoder %in% c("viterbi", "greedy")))
    stop(sprintf("benRea: decoder must be viterbi or greedy, got %r", decoder))
  labels <- bio_labels(types)
  if (decoder == "viterbi") {
    res <- viterbi_decode(emissions, labels, transition_scores = transition_scores)
    path <- res$path; score <- res$score
  } else {
    path <- greedy_decode(emissions, labels)
    score <- sum(sapply(seq_along(path), function(t) emissions[[t]][match(path[[t]], labels)]))
  }
  spans <- extract_spans(path)
  payload <- list(estimate = path, path = path, score = score,
                  spans = spans, valid = is_valid_bio(path),
                  labels = labels, decoder = decoder,
                  n_tokens = length(emissions), n_spans = length(spans),
                  method = "BIO named-entity decoding, Ramshaw & Marcus (1995) scheme, Viterbi (1967) constrained decoding")
  if (!is.null(gold))
    payload <- c(payload, span_f1(path, as.list(gold)))
  payload
}

nerdecode <- ner_decode
named_entity <- ner_decode
namedentity <- ner_decode

morie_benRea <- function(...) ner_decode(...)
