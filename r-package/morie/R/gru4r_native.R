# GRU4Rec: session-based recommendation with a ranking loss.
# Sources: Hidasi, B., Karatzoglou, A., Baltrunas, L. & Tikk, D.
# (2016) Session-based Recommendations with Recurrent Neural Networks,
# ICLR 2016, arXiv:1511.06939 -- session-parallel mini-batches with
# hidden-state reset (Sec. 3.1), the BPR and TOP1 ranking losses
# (Sec. 3.1.3, including the load-bearing sigma(r_j^2) regulariser
# in TOP1), and the single-GRU finding (Sec. 4); Rendle, S. et al.
# (2009) BPR: Bayesian Personalized Ranking from Implicit Feedback,
# UAI 2009 -- the BPR loss reused here; Cho, K. et al. (2014)
# Learning Phrase Representations using RNN Encoder-Decoder -- the GRU
# unit.
#
# Native R port mirroring morie.fn.gru4rec exactly. The Python arm
# uses math.log with a small floor to guard against log(0); the same
# floor is used here, and the same sigmoid (clamped below -700 for
# numerical stability) is used everywhere a probability is needed.

#' Numerically stable sigmoid: below -700 it is effectively 0
#'
#' A step of the gru4r_native implementation. Called by \code{morie_gru4r_bpr}, \code{morie_gru4r_gru}, \code{morie_gru4r_top1}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x Numeric; combined arithmetically in the body.
#' @return The value of \code{ifelse}.
#' @export
.gru4r_sigmoid <- function(x) {
  # Numerically stable sigmoid: below -700 it is effectively 0.
  ifelse(x > -700, 1 / (1 + exp(-x)), 0)
}

#' Session-parallel mini-batches
#'
#' Hidasi et al. (2016) Sec. 3.1. Slot \code{b} holds one session;
#' when that session runs out the next free session takes the slot
#' and the slot is flagged for a hidden-state reset, because
#' sessions are assumed independent.
#'
#' @param sessions List of integer vectors (length >= 2 each).
#' @param batch_size Number of parallel slots, between 1 and the
#'   number of sessions.
#' @return A list with \code{steps} (one element per batch tick:
#'   \code{input}, \code{target}, \code{reset}), \code{n_steps},
#'   \code{batch_size}, \code{n_sessions}, \code{note}.
#' @references Hidasi, B. et al. (2016). Session-based Recommendations
#'   with Recurrent Neural Networks. ICLR 2016.
#' @export
morie_gru4r <- function(sessions, batch_size) {
  S <- lapply(sessions, as.integer)
  if (any(vapply(S, length, integer(1)) < 2L))
    stop("gru4r: every session needs at least 2 events")
  B <- as.integer(batch_size)
  if (B < 1L || B > length(S))
    stop("gru4r: batch_size must lie in 1..", length(S),
         ", got ", B)
  slot <- as.list(seq_len(B) - 1L)
  pos <- rep(0L, B)
  nxt <- B
  steps <- list()
  repeat {
    x <- vector("list", B); y <- vector("list", B)
    reset <- rep(FALSE, B)
    alive <- FALSE
    for (b in seq_len(B)) {
      if (is.null(slot[[b]])) next
      s <- S[[slot[[b]] + 1L]]
      if (pos[b] + 1L >= length(s)) {
        if (nxt < length(S)) {
          slot[[b]] <- nxt; pos[b] <- 0L; nxt <- nxt + 1L
          reset[b] <- TRUE
          s <- S[[slot[[b]] + 1L]]
        } else {
          slot[[b]] <- NULL
          next
        }
      }
      x[[b]] <- s[pos[b] + 1L]
      y[[b]] <- s[pos[b] + 2L]
      pos[b] <- pos[b] + 1L
      alive <- TRUE
    }
    if (!alive) break
    steps[[length(steps) + 1L]] <- list(input = x, target = y,
                                        reset = reset)
  }
  list(steps = steps, n_steps = length(steps), batch_size = B,
       n_sessions = length(S),
       note = paste0("a slot's hidden state is reset when a new ",
                     "session takes it, because sessions are assumed ",
                     "independent"))
}

#' BPR ranking loss
#'
#' \eqn{-1/N_S \\sum_j \\log\\sigma(r_i - r_j)}, with the
#' log floored at 1e-12 for numerical stability.
#'
#' @param r_target Score of the target item.
#' @param r_negatives Numeric vector of negative-item scores.
#' @return Scalar loss.
#' @export
morie_gru4r_bpr <- function(r_target, r_negatives) {
  neg <- as.numeric(r_negatives)
  if (length(neg) == 0L) stop("gru4r: at least one negative is needed")
  rt <- as.numeric(r_target)
  s <- .gru4r_sigmoid(rt - neg)
  -sum(log(pmax(s, 1e-12))) / length(neg)
}

#' TOP1 ranking loss
#'
#' The smoothed relative rank
#' \eqn{\\sigma(r_j - r_i)} plus, when \code{regularize=TRUE}, the
#' load-bearing \eqn{\\sigma(r_j^2)} term that stops scores from
#' running away.
#'
#' @param r_target Score of the target item.
#' @param r_negatives Numeric vector of negative-item scores.
#' @param regularize Include the sigma(r_j^2) regulariser.
#' @return Scalar loss.
#' @export
morie_gru4r_top1 <- function(r_target, r_negatives, regularize = TRUE) {
  neg <- as.numeric(r_negatives)
  if (length(neg) == 0L) stop("gru4r: at least one negative is needed")
  rt <- as.numeric(r_target)
  rank <- sum(.gru4r_sigmoid(neg - rt)) / length(neg)
  if (!regularize) return(rank)
  rank + sum(.gru4r_sigmoid(neg * neg)) / length(neg)
}

#' One GRU update (single layer)
#'
#' Update gate, reset gate, candidate, and the convex combination
#' \code{(1 - z) h + z hh} that defines a GRU.
#'
#' @param x Input vector (length d).
#' @param h Hidden state (length n).
#' @param Wz,Uz Update-gate linear maps.
#' @param Wr,Ur Reset-gate linear maps.
#' @param Wh,Uh Candidate-h linear maps.
#' @return New hidden state.
#' @export
morie_gru4r_gru <- function(x, h, Wz, Uz, Wr, Ur, Wh, Uh) {
  x <- as.numeric(x); h <- as.numeric(h)
  Wz <- as.matrix(Wz); Uz <- as.matrix(Uz)
  Wr <- as.matrix(Wr); Ur <- as.matrix(Ur)
  Wh <- as.matrix(Wh); Uh <- as.matrix(Uh)
  n <- length(h)
  lin <- function(W, U, xv, hv) as.numeric(W %*% xv + U %*% hv)
  z <- .gru4r_sigmoid(lin(Wz, Uz, x, h))
  r <- .gru4r_sigmoid(lin(Wr, Ur, x, h))
  hh <- tanh(lin(Wh, Uh, x, r * h))
  (1 - z) * h + z * hh
}

#' Recall at k
#'
#' 1 if the target is in the top \code{k} ranked items, else 0.
#'
#' @param ranked Ordered integer vector of recommended item ids.
#' @param target Target item id.
#' @param kk Cutoff.
#' @return 0 or 1.
#' @export
morie_gru4r_recall <- function(ranked, target, kk = 20) {
  top <- as.integer(ranked)[seq_len(min(as.integer(kk),
                                        length(ranked)))]
  if (as.integer(target) %in% top) 1.0 else 0.0
}

#' Mean reciprocal rank at k
#'
#' 1 / position of the target within the top k, or 0 if absent.
#'
#' @param ranked Ordered integer vector of recommended item ids.
#' @param target Target item id.
#' @param kk Cutoff.
#' @return Scalar.
#' @export
morie_gru4r_mrr <- function(ranked, target, kk = 20) {
  top <- as.integer(ranked)[seq_len(min(as.integer(kk),
                                        length(ranked)))]
  t <- as.integer(target)
  if (t %in% top) 1.0 / (which(top == t)[1L]) else 0.0
}
