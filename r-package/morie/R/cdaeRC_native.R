# Collaborative denoising auto-encoder for top-N recommendation.
# Sources: Wu, Y., DuBois, C., Zheng, A. X. & Ester, M. (2016)
# "Collaborative Denoising Auto-Encoders for Top-N Recommender
# Systems", *Proceedings of the Ninth ACM International Conference
# on Web Search and Data Mining (WSDM '16)*, 153-162,
# doi:10.1145/2835776.2835837, for Sec. 2's point-wise and
# pair-wise objective framework, the four loss functions of Table 1
# and the warning that log and hinge losses need y = -1 for
# negatives, and the need to augment positives with sampled
# negatives; Sec. 2.3 for the auto-encoder, the tied weights and
# the mask-out/drop-out corruption scaled by 1/(1-q) so it stays
# unbiased; Sec. 3 (eqs. (9)-(13)) for the CDAE with the
# user-specific node V_u, and Algorithm 1 for the SGD with
# negative sampling. Vincent, P., Larochelle, H., Bengio, Y. &
# Manzagol, P.-A. (2008) "Extracting and composing robust
# features with denoising autoencoders", *ICML 2008*, 1096-1103,
# doi:10.1145/1390156.1390294, for the denoising auto-encoder.
# Rendle, S., Freudenthaler, C., Gantner, Z. & Schmidt-Thieme, L.
# (2009) "BPR: Bayesian Personalized Ranking from Implicit
# Feedback", *UAI 2009*, 452-461, arXiv:1205.2618, for the
# pair-wise objective in Table 1.

.CDAE_EPS <- 1e-12
.CDAE_LOSSES <- c("square", "log", "hinge", "cross_entropy")
.CDAE_ACTS <- c("sigmoid", "identity", "tanh")

.cdae_act <- function(name, x) {
  if (name == "sigmoid") {
    if (x >= -700) 1.0 / (1.0 + exp(-x)) else 0.0
  } else if (name == "identity") {
    x
  } else if (name == "tanh") {
    tanh(x)
  } else {
    stop(sprintf("cdaeRC: activation must be one of %s, got '%s'",
                 paste(.CDAE_ACTS, collapse = ", "), name))
  }
}

.cdae_dact <- function(name, y) {
  if (name == "sigmoid") y * (1.0 - y)
  else if (name == "identity") 1.0
  else 1.0 - y * y
}

#' corrupt
#'
#' Part of the cdaeRC_native implementation; see the file header for the
#' source it follows.
#'
#' @param y See Usage.
#' @param q See Usage.
#' @param e See Usage.
#' @return The value of \code{ifelse}.
#' @export
corrupt <- function(y, q, e) {
  qq <- as.numeric(q)
  if (is.na(qq) || qq < 0 || qq >= 1)
    stop(sprintf("cdaeRC: q must lie in [0,1), got %r", q))
  d <- 1.0 / (1.0 - qq)
  u <- .ghc_unif(e, length(y))
  ifelse(u < qq, 0.0, d * as.numeric(y))
}

#' encode
#'
#' Part of the cdaeRC_native implementation; see the file header for the
#' source it follows.
#'
#' @param y_tilde See Usage.
#' @param W See Usage.
#' @param V_u See Usage.
#' @param b See Usage.
#' @param activation Defaults to \code{"sigmoid"}.
#' @return The value of \code{z}, as built in the body.
#' @export
encode <- function(y_tilde, W, V_u, b, activation = "sigmoid") {
  K <- length(b)
  z <- numeric(K)
  for (f in seq_len(K)) {
    s <- b[f] + V_u[f]
    for (i in seq_along(y_tilde))
      if (y_tilde[i] != 0.0)
        s <- s + W[i, f] * y_tilde[i]
    z[f] <- .cdae_act(activation, s)
  }
  z
}

#' morie_cdaeRC_decode
#'
#' Part of the cdaeRC_native implementation; see the file header for the
#' source it follows.
#'
#' @param z See Usage.
#' @param Wp See Usage.
#' @param bp See Usage.
#' @param items Defaults to \code{NULL}.
#' @param activation Defaults to \code{"sigmoid"}.
#' @return The value of \code{out}, as built in the body.
#' @export
morie_cdaeRC_decode <- function(z, Wp, bp, items = NULL, activation = "sigmoid") {
  idx <- if (is.null(items)) seq_along(bp) else as.integer(items)
  out <- numeric(length(idx))
  names(out) <- as.character(idx)
  for (j in seq_along(idx)) {
    i <- idx[j]
    s <- bp[i]
    for (f in seq_along(z))
      s <- s + Wp[i, f] * z[f]
    out[j] <- .cdae_act(activation, s)
  }
  out
}

#' loss
#'
#' Part of the cdaeRC_native implementation; see the file header for the
#' source it follows.
#'
#' @param y See Usage.
#' @param y_hat See Usage.
#' @param kind Defaults to \code{"square"}.
#' @return One of two values, depending on the branch taken.
#' @export
loss <- function(y, y_hat, kind = "square") {
  if (!(kind %in% .CDAE_LOSSES))
    stop(sprintf("cdaeRC: loss must be one of %s, got '%s'",
                 paste(.CDAE_LOSSES, collapse = ", "), kind))
  yv <- as.numeric(y); yh <- as.numeric(y_hat)
  if (kind %in% c("log", "hinge") && yv == 0.0)
    stop(sprintf("cdaeRC: the %s loss needs y = -1 for negatives, not 0", kind))
  if (kind == "square")
    return(0.5 * (yv - yh) ^ 2)
  if (kind == "log") {
    if (-yv * yh < 700) log(1.0 + exp(-yv * yh)) else -yv * yh
  } else if (kind == "hinge") {
    max(0.0, 1.0 - yv * yh)
  } else {
    p <- if (yh >= -700) 1.0 / (1.0 + exp(-yh)) else 0.0
    p <- min(max(p, .CDAE_EPS), 1.0 - .CDAE_EPS)
    -yv * log(p) - (1.0 - yv) * log(1.0 - p)
  }
}

#' fit_cdae
#'
#' Part of the cdaeRC_native implementation; see the file header for the
#' source it follows.
#'
#' @param pos See Usage.
#' @param n_users See Usage.
#' @param n_items See Usage.
#' @param k_dim Defaults to \code{8L}.
#' @param q Defaults to \code{0.2}.
#' @param alpha Defaults to \code{0.05}.
#' @param lam Defaults to \code{0.01}.
#' @param iters Defaults to \code{30L}.
#' @param n_neg Defaults to \code{5L}.
#' @param seed Defaults to \code{0}.
#' @param activation Defaults to \code{"sigmoid"}.
#' @param init_scale Defaults to \code{0.1}.
#' @return A list with \code{estimate}, \code{W}, \code{W_prime}, \code{V}, \code{b}, \code{b_prime}, \code{loss_history}, \code{final_loss}, \code{k}, \code{q}, \code{n_neg}, \code{activation}, \code{method}, \code{note}.
#' @export
fit_cdae <- function(pos, n_users, n_items, k_dim = 8L, q = 0.2,
                     alpha = 0.05, lam = 0.01, iters = 30L,
                     n_neg = 5L, seed = 0, activation = "sigmoid",
                     init_scale = 0.1) {
  U <- as.integer(n_users); I <- as.integer(n_items); K <- as.integer(k_dim)
  if (U < 1L || I < 2L || K < 1L)
    stop("cdaeRC: need at least 1 user, 2 items and 1 hidden node")
  e <- .ghc_rng(as.numeric(seed))

  rand <- function() (.ghc_unif(e, 1L) - 0.5) * 2.0 * init_scale

  W  <- matrix(0.0, nrow = I, ncol = K)
  Wp <- matrix(0.0, nrow = I, ncol = K)
  V  <- matrix(0.0, nrow = U, ncol = K)
  b  <- numeric(K)
  bp <- numeric(I)
  for (i in seq_len(I)) for (f in seq_len(K)) { W[i, f]  <- rand(); Wp[i, f] <- rand() }
  for (u in seq_len(U)) for (f in seq_len(K)) V[u, f] <- rand()

  a  <- as.numeric(alpha); lm <- as.numeric(lam)
  hist <- numeric(as.integer(iters))
  qq <- as.numeric(q)
  if (is.na(qq) || qq < 0 || qq >= 1)
    stop(sprintf("cdaeRC: q must lie in [0,1), got %r", q))

  for (it in seq_len(as.integer(iters))) {
    tot <- 0.0
    for (u in seq_len(U)) {
      seen <- if (is.null(pos[[as.character(u)]])) integer(0) else
                sort(unique(as.integer(pos[[as.character(u)]])))
      if (length(seen) == 0L) next
      seen_set <- as.character(seen)
      y <- ifelse(as.character(seq_len(I) - 1L) %in% seen_set, 1.0, 0.0)
      yt <- corrupt(y, qq, e)
      z <- encode(yt, W, V[u, ], b, activation)
      neg <- integer(0)
      guard <- 0L
      n_neg_i <- as.integer(n_neg)
      while (length(neg) < n_neg_i && guard < 100L * n_neg_i) {
        j <- as.integer(floor(.ghc_unif(e, 1L) * I)) %% I
        if (!(as.character(j) %in% seen_set))
          neg <- c(neg, j)
        guard <- guard + 1L
      }
      tgt <- c(seen, neg)
      yi_tgt <- ifelse(as.character(tgt) %in% seen_set, 1.0, 0.0)
      out_v <- numeric(length(tgt))
      for (k in seq_along(tgt)) {
        i <- tgt[k] + 1L
        s <- bp[i]
        for (f in seq_len(K)) s <- s + Wp[i, f] * z[f]
        out_v[k] <- .cdae_act(activation, s)
      }
      dz <- numeric(K)
      for (k in seq_along(tgt)) {
        i <- tgt[k] + 1L
        yi <- yi_tgt[k]
        e_i <- (out_v[k] - yi) * .cdae_dact(activation, out_v[k])
        tot <- tot + loss(yi, out_v[k], "square")
        for (f in seq_len(K)) {
          dz[f] <- dz[f] + e_i * Wp[i, f]
          Wp[i, f] <- Wp[i, f] - a * (e_i * z[f] + lm * Wp[i, f])
        }
        bp[i] <- bp[i] - a * e_i
      }
      dpre <- dz * .cdae_dact(activation, z)
      for (i in seq_len(I)) {
        if (yt[i] != 0.0) {
          for (f in seq_len(K))
            W[i, f] <- W[i, f] - a * (dpre[f] * yt[i] + lm * W[i, f])
        }
      }
      for (f in seq_len(K)) {
        V[u, f] <- V[u, f] - a * (dpre[f] + lm * V[u, f])
        b[f]    <- b[f]    - a * dpre[f]
      }
    }
    hist[it] <- tot
  }
  list(estimate = list(W = W, W_prime = Wp, V = V, b = b, b_prime = bp),
       W = W, W_prime = Wp, V = V, b = b, b_prime = bp,
       loss_history = hist,
       final_loss = if (length(hist)) hist[length(hist)] else NaN,
       k = K, q = as.numeric(qq), n_neg = as.integer(n_neg),
       activation = activation,
       method = "CDAE; Wu, DuBois, Zheng & Ester (2016) eqs. (9)-(13), Algorithm 1",
       note = "V_u is the user-specific input node -- without it this is an ordinary denoising auto-encoder over item vectors")
}

#' recommend
#'
#' Part of the cdaeRC_native implementation; see the file header for the
#' source it follows.
#'
#' @param model See Usage.
#' @param pos See Usage.
#' @param u See Usage.
#' @param n_items See Usage.
#' @param top_k Defaults to \code{5L}.
#' @param activation Defaults to \code{"sigmoid"}.
#' @return A list with \code{ranking}, \code{n_scored}.
#' @export
recommend <- function(model, pos, u, n_items, top_k = 5L,
                      activation = "sigmoid") {
  W <- model$W; Wp <- model$W_prime; V <- model$V
  b <- model$b; bp <- model$b_prime
  u <- as.integer(u)
  seen <- if (is.null(pos[[as.character(u)]])) integer(0) else
            sort(unique(as.integer(pos[[as.character(u)]])))
  seen_set <- as.character(seen)
  I <- as.integer(n_items)
  y <- ifelse(as.character(seq_len(I) - 1L) %in% seen_set, 1.0, 0.0)
  z <- encode(y, W, V[u, ], b, activation)
  s <- numeric(I)
  for (i in seq_len(I)) {
    ss <- bp[i]
    K <- length(z)
    for (f in seq_len(K)) ss <- ss + Wp[i, f] * z[f]
    s[i] <- .cdae_act(activation, ss)
  }
  cand <- which(!(as.character(seq_len(I) - 1L) %in% seen_set))
  sc <- s[cand + 0L]
  ord <- order(-sc)
  top <- head(ord, as.integer(top_k))
  list(ranking = lapply(cand[top], function(i) list(item = i - 1L, score = s[i])),
       n_scored = length(cand))
}

#' morie_cdaeRC
#'
#' Part of the cdaeRC_native implementation; see the file header for the
#' source it follows.
#'
#' @param pos See Usage.
#' @param n_users See Usage.
#' @param n_items See Usage.
#' @param k_dim Defaults to \code{8L}.
#' @param q Defaults to \code{0.2}.
#' @param alpha Defaults to \code{0.05}.
#' @param lam Defaults to \code{0.01}.
#' @param iters Defaults to \code{30L}.
#' @param n_neg Defaults to \code{5L}.
#' @param seed Defaults to \code{0}.
#' @param activation Defaults to \code{"sigmoid"}.
#' @param init_scale Defaults to \code{0.1}.
#' @return The value of \code{fit_cdae}.
#' @export
morie_cdaeRC <- function(pos, n_users, n_items, k_dim = 8L, q = 0.2,
                         alpha = 0.05, lam = 0.01, iters = 30L,
                         n_neg = 5L, seed = 0, activation = "sigmoid",
                         init_scale = 0.1) {
  fit_cdae(pos, n_users, n_items, k_dim, q, alpha, lam, iters,
           n_neg, seed, activation, init_scale)
}

cdae <- fit_cdae
collaborativedenoisingautoencoder <- fit_cdae

.cdaeRC_cheatsheet <- function() {
  paste("cdaeRC: a denoising auto-encoder over a user's BINARY",
        "preference vector, plus a USER-SPECIFIC input node V_u --",
        "that node is what separates it from a plain DAE and makes",
        "W_i, V_u item and user embeddings. Corruption is",
        "mask-out with probability q, survivors scaled by",
        "1/(1-q) so the corruption is UNBIASED. Positives only",
        "would train the all-ones model, so negatives are SAMPLED.",
        "Four losses offered; log and hinge need the negative",
        "label to be -1, not 0.")
}
