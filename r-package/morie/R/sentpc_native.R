# SentencePiece: subword tokenisation that is losslessly reversible.
# Sources: Kudo, T. & Richardson, J. (2018) "SentencePiece: A simple
# and language independent subword tokenizer and detokenizer for
# Neural Text Processing", EMNLP 2018 System Demonstrations, 66-71,
# doi:10.18653/v1/D18-2012 (lossless encoding with U+2581 for
# whitespace, BPE and unigram, fixed vocabulary size); Sennrich, R.,
# Haddow, B. & Birch, A. (2016) ACL 1715-1725 (BPE merges); Kudo, T.
# (2018) ACL 66-75 (unigram LM and Viterbi).
#
# Native R arm mirroring the Python arm exactly: the same U+2581
# escaping and the same _units split that keeps every unit starting
# at the marker so runs of spaces do not collapse, the same greedy
# BPE trainer that merges the most frequent pair until the
# vocabulary is full, the same Viterbi over the split lattice for
# the unigram model, and morie_sentpc_decode/unescape as a pure string operation.

.EPS <- 1e-300
.SPACE <- "\u2581"

.escape_whitespace <- function(text, add_prefix = TRUE) {
  s <- as.character(text)
  out <- gsub(" ", .SPACE, s, fixed = TRUE)
  if (isTRUE(add_prefix)) out <- paste0(.SPACE, out)
  out
}

.unescape_whitespace <- function(text, strip_prefix = TRUE) {
  s <- as.character(text)
  if (isTRUE(strip_prefix) && startsWith(s, .SPACE))
    s <- substring(s, 2L)
  gsub(.SPACE, " ", s, fixed = TRUE)
}

# Split an escaped string so every unit after the first begins with
# U+2581; joining reproduces the input exactly so runs of spaces do
# not collapse.
.units <- function(escaped) {
  out <- character(0); cur <- ""
  chars <- strsplit(escaped, "")[[1]]
  for (ch in chars) {
    if (ch == .SPACE) {
      if (nzchar(cur)) out <- c(out, cur)
      cur <- .SPACE
    } else {
      cur <- paste0(cur, ch)
    }
  }
  if (nzchar(cur)) out <- c(out, cur)
  out
}

#' Escape whitespace as U+2581, optionally prefixing the marker
#' @export
escape_whitespace <- function(text, add_prefix = TRUE) {
  .escape_whitespace(text, add_prefix)
}

#' Invert the whitespace escape
#' @export
unescape_whitespace <- function(text, strip_prefix = TRUE) {
  .unescape_whitespace(text, strip_prefix)
}

#' Decode pieces back to text (pure string operation)
#' @export
morie_sentpc_decode <- function(pieces, strip_prefix = TRUE) {
  .unescape_whitespace(paste0(as.character(pieces), collapse = ""),
                       strip_prefix)
}

#' Train a BPE model: merge the most frequent adjacent pair greedily
#' @export
train_bpe <- function(corpus, vocab_size, add_prefix = TRUE) {
  V <- as.integer(vocab_size)
  if (V < 1L) stop("sentpc: vocab_size must be at least 1")
  words <- list()
  for (line in corpus) {
    for (w in .units(.escape_whitespace(line, add_prefix))) {
      key <- w
      if (is.null(words[[key]])) words[[key]] <- 0L
      words[[key]] <- words[[key]] + 1L
    }
  }
  if (length(words) == 0L) {
    stop("sentpc: the corpus produced no tokens")
  }
  alphabet <- sort(unique(unlist(strsplit(unlist(names(words)), ""))))
  merges <- list()
  vocab <- alphabet
  while (length(vocab) < V) {
    pairs <- list()
    for (key in names(words)) {
      w <- strsplit(key, "")[[1]]
      f <- words[[key]]
      if (length(w) < 2L) next
      for (i in seq_len(length(w) - 1L)) {
        pk <- paste0(w[i], "|", w[i + 1L])
        if (is.null(pairs[[pk]])) pairs[[pk]] <- 0L
        pairs[[pk]] <- pairs[[pk]] + f
      }
    }
    if (length(pairs) == 0L) break
    cnts <- unlist(pairs)
    best_key <- names(which.max(cnts))
    best <- strsplit(best_key, "|", fixed = TRUE)[[1]]
    merges[[length(merges) + 1L]] <- best
    new_token <- paste0(best[1L], best[2L])
    vocab <- c(vocab, new_token)
    nw <- list()
    for (key in names(words)) {
      w <- strsplit(key, "")[[1]]
      f <- words[[key]]
      out <- character(0); i <- 1L
      while (i <= length(w)) {
        if (i < length(w) && w[i] == best[1L] && w[i + 1L] == best[2L]) {
          out <- c(out, paste0(w[i], w[i + 1L]))
          i <- i + 2L
        } else {
          out <- c(out, w[i])
          i <- i + 1L
        }
      }
      nk <- paste0(out, collapse = "")
      if (is.null(nw[[nk]])) nw[[nk]] <- 0L
      nw[[nk]] <- nw[[nk]] + f
    }
    words <- nw
  }
  list(merges = merges, vocab = sort(vocab),
       vocab_size = length(vocab), requested = V,
       algorithm = "bpe",
       note = paste("greedy and deterministic -- the merge list fixes ",
                    "every later segmentation"))
}

#' Encode text with a trained BPE model
#' @export
encode_bpe <- function(text, model, add_prefix = TRUE) {
  esc <- .escape_whitespace(text, add_prefix)
  out <- character(0)
  for (w in .units(esc)) {
    toks <- strsplit(w, "")[[1]]
    for (m in model$merges) {
      a <- m[1L]; b <- m[2L]
      i <- 1L; neww <- character(0)
      while (i <= length(toks)) {
        if (i < length(toks) && toks[i] == a && toks[i + 1L] == b) {
          neww <- c(neww, paste0(toks[i], toks[i + 1L]))
          i <- i + 2L
        } else {
          neww <- c(neww, toks[i])
          i <- i + 1L
        }
      }
      toks <- neww
    }
    out <- c(out, toks)
  }
  out
}

#' Unigram LM segmentation by Viterbi over the split lattice
#' @export
viterbi_segment <- function(text, piece_logp, add_prefix = TRUE) {
  s <- .escape_whitespace(text, add_prefix)
  n <- nchar(s)
  if (nchar(s) == 0L) return(list(pieces = character(0), logp = 0))
  if (length(piece_logp) == 0L) maxlen <- 1L
  else maxlen <- max(nchar(names(piece_logp)))
  best <- rep(-Inf, n + 1L)
  back <- vector("list", n + 1L)
  best[1L] <- 0
  for (i in 2:(n + 1L)) {
    for (L in 1:min(maxlen, i - 1L)) {
      piece <- substr(s, i - L, i - 1L)
      lp <- piece_logp[[piece]]
      if (is.null(lp)) next
      if (best[i - L] + lp > best[i]) {
        best[i] <- best[i - L] + lp
        back[[i]] <- c(i - L, piece)
      }
    }
  }
  if (is.infinite(best[n + 1L]) && best[n + 1L] < 0) {
    stop("sentpc: no segmentation covers the input -- the piece set ",
         "must include every character")
  }
  pieces <- character(0); i <- n + 1L
  while (i > 1L) {
    st <- back[[i]]
    pieces <- c(st[2L], pieces)
    i <- as.integer(st[1L])
  }
  list(pieces = pieces, logp = best[n + 1L],
       n_pieces = length(pieces),
       algorithm = "unigram (Viterbi)")
}

# house entry point: the package exports one morie_<module>
morie_sentpc <- viterbi_segment
