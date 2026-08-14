# morie.fn -- function file (rootcoder007/morie)
# ELMo: deep contextualized word representations.
#
# References
# Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K. &
# Zettlemoyer, L. (2018) "Deep contextualized word representations",
# Proceedings of NAACL-HLT 2018, 2227-2237, arXiv:1802.05365. Sec. 3,
# eq. (1), and the layer-weighting scheme.
# Hochreiter, S. & Schmidhuber, J. (1997) "Long Short-Term Memory",
# Neural Computation 9(8), 1735-1780. The recurrent cell the biLM is
# built from.
# Ba, J. L., Kiros, J. R. & Hinton, G. E. (2016) "Layer Normalization",
# arXiv:1607.06450. The normalisation the paper applies per layer
# before weighting.

.EPS <- 1e-12

.sigmoid <- function(x) 1 / (1 + exp(-x))

.layer_weights <- function(raw) {
  if (length(raw) == 0L) stop("elmo: no layer weights given")
  mx <- max(raw)
  e <- exp(as.numeric(raw) - mx)
  tot <- sum(e)
  as.numeric(e / tot)
}

.lstm_step <- function(x, h, c, Wx, Wh, b) {
  d <- length(h)
  if (length(c) != d) stop("elmo: hidden and cell sizes differ")
  xa <- as.numeric(x)
  ha <- as.numeric(h)
  ca <- as.numeric(c)
  Wxa <- as.matrix(Wx)
  Wha <- as.matrix(Wh)
  ba <- as.numeric(b)
  z <- as.numeric(Wxa %*% xa + Wha %*% ha + ba)
  ig <- .sigmoid(z[seq_len(d)])
  fg <- .sigmoid(z[d + seq_len(d)])
  gg <- tanh(z[2 * d + seq_len(d)])
  og <- .sigmoid(z[3 * d + seq_len(d)])
  cn <- fg * ca + ig * gg
  hn <- og * tanh(cn)
  list(h = hn, c = cn)
}

.bilm_forward <- function(X, layers) {
  Xm <- as.matrix(X)
  L <- nrow(Xm)
  if (L == 0L) stop("elmo: empty sequence")
  reps <- vector("list", length(layers) + 1L)
  # Layer 0 is the token representation DUPLICATED, h_{k,0} = [x_k; x_k]
  xdup <- cbind(Xm, Xm)
  reps[[1L]] <- xdup
  cur <- Xm
  for (li in seq_along(layers)) {
    lyr <- layers[[li]]
    Wxf <- as.matrix(lyr[[1L]]); Whf <- as.matrix(lyr[[2L]]); bf <- as.numeric(lyr[[3L]])
    Wxb <- as.matrix(lyr[[4L]]); Whb <- as.matrix(lyr[[5L]]); bb <- as.numeric(lyr[[6L]])
    d <- ncol(Whf)
    if (ncol(reps[[1L]]) != 2L * d) {
      stop(sprintf("elmo: token dimension %d but hidden dimension %d; layer 0 is [x; x] so they must match",
                   ncol(Xm), d))
    }
    h <- rep(0, d); c <- rep(0, d)
    fwd <- matrix(0, nrow = L, ncol = d)
    for (t in seq_len(L)) {
      r <- .lstm_step(cur[t, , drop = FALSE], h, c, Wxf, Whf, bf)
      h <- r$h; c <- r$c
      fwd[t, ] <- h
    }
    h <- rep(0, d); c <- rep(0, d)
    bwd <- matrix(0, nrow = L, ncol = d)
    for (t in L:1L) {
      r <- .lstm_step(cur[t, , drop = FALSE], h, c, Wxb, Whb, bb)
      h <- r$h; c <- r$c
      bwd[t, ] <- h
    }
    cur <- cbind(fwd, bwd)
    reps[[li + 1L]] <- cur
  }
  reps
}

.elmo_mix <- function(reps, raw_weights, gamma = 1.0, position = NULL) {
  n_layers <- length(reps)
  if (length(raw_weights) != n_layers) {
    stop(sprintf("elmo: %d weights for %d layers", length(raw_weights), n_layers))
  }
  s <- .layer_weights(raw_weights)
  L <- nrow(reps[[1L]])
  dims <- unique(vapply(reps, ncol, integer(1)))
  if (length(dims) != 1L) {
    stop(sprintf("elmo: layers have differing widths %s",
                 paste(sort(dims), collapse = ",")))
  }
  d <- dims
  idx <- if (is.null(position)) seq_len(L) else as.integer(position)
  out <- matrix(0, nrow = length(idx), ncol = d)
  for (ti in seq_along(idx)) {
    t <- idx[ti]
    for (j in seq_len(n_layers)) {
      for (cc in seq_len(d)) {
        out[ti, cc] <- out[ti, cc] + gamma * s[j] * reps[[j]][t, cc]
      }
    }
  }
  if (is.null(position)) out else out[1L, , drop = TRUE]
}

.elmo_representation <- function(X, layers, raw_weights = NULL, gamma = 1.0) {
  reps <- .bilm_forward(X, layers)
  n <- length(reps)
  raw <- if (is.null(raw_weights)) rep(0, n) else as.numeric(raw_weights)
  mixed <- .elmo_mix(reps, raw, gamma = gamma)
  s <- .layer_weights(raw)
  list(
    estimate = mixed, elmo = mixed, layers = reps, weights = s,
    gamma = as.numeric(gamma), n_layers = n, L = nrow(reps[[1L]]),
    d = if (length(mixed) > 0L) ncol(mixed) else 0L,
    top_layer = reps[[n]],
    method = "ELMo layer mixture, Peters et al. (2018) eq. (1)"
  )
}

morie_elmo <- function(X, layers, raw_weights = NULL, gamma = 1.0) {
  .elmo_representation(X, layers, raw_weights = raw_weights, gamma = gamma)
}
