# SPDX-License-Identifier: AGPL-3.0-or-later
#' ViT fine-tune for a downstream task
#'
#' Dosovitskiy et al. (2021), \emph{An Image is Worth 16x16 Words}, ICLR 2021,
#' arXiv:2010.11929v2, Section 3.2, p. 4: "we remove the pre-trained prediction
#' head and attach a zero-initialized D x K feedforward layer, where K is the
#' number of downstream classes", and Section 3.1, p. 3: the head sits on
#' z_L^0, the representation y of Eq. (4).  Section B.1.1, p. 13, fine-tunes
#' with SGD and momentum 0.9; this arm uses plain full-batch gradient descent
#' instead, with a fixed step count, because both language arms must land on
#' the same numbers and momentum adds nothing a determinism check would notice.
#' That substitution is an implementation choice, not the paper's recipe.
#'
#' mode is the frozen/unfrozen distinction of Section 3.2 reduced to what this
#' API actually holds -- representations, not a network: "linear" trains the
#' head alone; "full" also trains a per-feature affine recalibration of the
#' representation, standing in for unfreezing the backbone.  It cannot be
#' back-propagation into a backbone this function was never given.
#'
#' @param model n-by-D matrix of representations, one row per image.
#' @param data Length-n integer class labels in 1 ... K.
#' @param mode "linear" or "full".
#' @param steps Gradient-descent steps.
#' @param lr Step size.
#' @param eps Guard inside the log of the cross-entropy.
#' @return list: estimate (training accuracy), predictions, probs, loss, W, b,
#'   gain, shift, n, n_classes, embed_dim, mode, method.
#' @keywords internal
#' @examples
#' Vitfsv(matrix(c(1, 0, 0, 1), 2, byrow = TRUE), c(1, 2), steps = 50)$predictions
#' @export
Vitfsv <- function(model, data, mode = "linear", steps = 200L, lr = 0.5,
                   eps = 1e-5) {
  X <- .s03mat(model)
  n <- nrow(X)
  if (n == 0L) stop("vit_finetune: model is empty")
  D <- ncol(X)
  y <- as.integer(.s03vec(data))
  if (length(y) != n) {
    stop("vit_finetune: data must have one label per row of model")
  }
  K <- max(y)
  if (min(y) < 1L) stop("vit_finetune: labels must be integers 1 ... K")
  if (K < 2L) stop("vit_finetune: need at least two classes")
  if (!(mode %in% c("linear", "full"))) {
    stop("vit_finetune: mode must be 'linear' or 'full'")
  }
  ns <- as.integer(steps)
  if (ns < 0L) stop("vit_finetune: steps must not be negative")
  W <- matrix(0, D, K); b <- numeric(K)
  a <- rep(1, D); cc <- numeric(D)
  full <- identical(mode, "full")
  loss <- 0
  P <- matrix(0, n, K)
  for (it in 0:ns) {
    Z <- matrix(0, n, D)
    for (i in seq_len(n)) {
      for (j in seq_len(D)) Z[i, j] <- a[j] * X[i, j] + cc[j]
    }
    loss <- 0
    for (i in seq_len(n)) {
      lg <- b
      for (j in seq_len(D)) {
        zij <- Z[i, j]
        for (t in seq_len(K)) lg[t] <- lg[t] + zij * W[j, t]
      }
      p <- .s03softmax(lg)
      P[i, ] <- p
      loss <- loss - log(p[y[i]] + eps)
    }
    loss <- loss / n
    if (it == ns) break
    gW <- matrix(0, D, K); gb <- numeric(K)
    ga <- numeric(D); gc <- numeric(D)
    for (i in seq_len(n)) {
      for (t in seq_len(K)) {
        d <- P[i, t] - (if (y[i] == t) 1 else 0)
        gb[t] <- gb[t] + d / n
        for (j in seq_len(D)) {
          gW[j, t] <- gW[j, t] + Z[i, j] * d / n
          if (full) {
            ga[j] <- ga[j] + d * W[j, t] * X[i, j] / n
            gc[j] <- gc[j] + d * W[j, t] / n
          }
        }
      }
    }
    W <- W - lr * gW
    b <- b - lr * gb
    if (full) {
      a <- a - lr * ga
      cc <- cc - lr * gc
    }
  }
  pred <- integer(n); hit <- 0L
  for (i in seq_len(n)) {
    best <- 1L
    for (t in seq_len(K)[-1L]) if (P[i, t] > P[i, best]) best <- t
    pred[i] <- best
    if (best == y[i]) hit <- hit + 1L
  }
  list(estimate = hit / n, predictions = pred, probs = P, loss = loss,
       W = W, b = b, gain = a, shift = cc, n = n, n_classes = K,
       embed_dim = D, mode = mode, method = "ViT fine-tune for downstream")
}
