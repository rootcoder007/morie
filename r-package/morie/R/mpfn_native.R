# mpfn -- message passing neural networks (Gilmer et al. 2017)
# References:
#   Gilmer et al. (2017) "Neural Message Passing for Quantum Chemistry" arXiv:1704.01212
#   Li et al. (2016) "Gated Graph Sequence Neural Networks" arXiv:1511.05493
#   Vinyals et al. (2016) "Order Matters: Sequence to sequence for sets" arXiv:1511.06391
# Base R only.

#' mpfn_sig
#'
#' Part of the mpfn_native implementation; see the file header for the
#' source it follows.
#'
#' @param x See Usage.
#' @return One of two values, depending on the branch taken.
#' @export
mpfn_sig <- function(x) {
  if (x > -700) 1 / (1 + exp(-x)) else 0
}

#' mpfn_message
#'
#' Part of the mpfn_native implementation; see the file header for the
#' source it follows.
#'
#' @param h_v See Usage.
#' @param h_w See Usage.
#' @param e_vw See Usage.
#' @param A Defaults to \code{NULL}.
#' @return A vector, from \code{as.numeric}.
#' @export
mpfn_message <- function(h_v, h_w, e_vw, A = NULL) {
  hw <- as.numeric(h_w)
  if (is.null(A)) {
    e <- if (is.numeric(e_vw) && length(e_vw) > 1L) e_vw[1] else as.numeric(e_vw)
    return(e * hw)
  }
  M <- A(e_vw)
  as.numeric(M %*% hw)
}

#' mpfn_update_gru
#'
#' Part of the mpfn_native implementation; see the file header for the
#' source it follows.
#'
#' @param h See Usage.
#' @param m See Usage.
#' @param Wz See Usage.
#' @param Uz See Usage.
#' @param Wr See Usage.
#' @param Ur See Usage.
#' @param Wh See Usage.
#' @param Uh See Usage.
#' @return A numeric value.
#' @export
mpfn_update_gru <- function(h, m, Wz, Uz, Wr, Ur, Wh, Uh) {
  n <- length(h)
  lin <- function(W, U, a, b) {
    as.numeric(W %*% a + U %*% b)
  }
  z <- sapply(lin(Wz, Uz, m, h), mpfn_sig)
  r <- sapply(lin(Wr, Ur, m, h), mpfn_sig)
  hh <- tanh(lin(Wh, Uh, m, r * h))
  (1 - z) * h + z * hh
}

#' mpfn_message_passing
#'
#' Part of the mpfn_native implementation; see the file header for the
#' source it follows.
#'
#' @param H0 See Usage.
#' @param adj See Usage.
#' @param edge_features See Usage.
#' @param T Defaults to \code{3}.
#' @param A Defaults to \code{NULL}.
#' @param update Defaults to \code{NULL}.
#' @return The value of \code{H}, as built in the body.
#' @export
mpfn_message_passing <- function(H0, adj, edge_features, T = 3, A = NULL,
                                 update = NULL) {
  H <- lapply(H0, as.numeric)
  if (as.integer(T) < 1L) stop("mpfn: T must be at least 1")
  for (step in seq_len(as.integer(T))) {
    new <- list()
    for (v in seq_along(H)) {
      nb <- if (!is.null(adj[[as.character(v)]])) adj[[as.character(v)]] else integer(0)
      m <- rep(0, length(H[[v]]))
      for (w in nb) {
        e <- edge_features[[paste(v, w, sep = ",")]]
        if (is.null(e)) e <- edge_features[[paste(w, v, sep = ",")]]
        if (is.null(e)) e <- 1
        mm <- mpfn_message(H[[v]], H[[w]], e, A)
        m <- m + mm
      }
      if (!is.null(update)) {
        new[[v]] <- update(H[[v]], m)
      } else {
        new[[v]] <- H[[v]] + m
      }
    }
    H <- new
  }
  H
}

#' mpfn_readout
#'
#' Part of the mpfn_native implementation; see the file header for the
#' source it follows.
#'
#' @param H See Usage.
#' @param how Defaults to \code{"sum"}.
#' @param H0 Defaults to \code{NULL}.
#' @param i_fn Defaults to \code{NULL}.
#' @param j_fn Defaults to \code{NULL}.
#' @return The value of \code{acc}, as built in the body.
#' @export
mpfn_readout <- function(H, how = "sum", H0 = NULL, i_fn = NULL, j_fn = NULL) {
  if (!(how %in% c("sum", "mean", "gated"))) {
    stop(sprintf("mpfn: readout must be one of sum, mean, gated, got %s", how))
  }
  rows <- lapply(H, as.numeric)
  d <- length(rows[[1]])
  if (how == "sum") {
    out <- rep(0, d)
    for (r in rows) out <- out + r
    return(out)
  }
  if (how == "mean") {
    out <- rep(0, d)
    for (r in rows) out <- out + r
    return(out / length(rows))
  }
  if (is.null(H0) || is.null(i_fn) || is.null(j_fn)) {
    stop("mpfn: the gated readout needs H0, i_fn and j_fn")
  }
  acc <- rep(0, d)
  for (v in seq_along(rows)) {
    g <- i_fn(rows[[v]], as.numeric(H0[[v]]))
    jv <- j_fn(rows[[v]])
    acc <- acc + sapply(g, mpfn_sig) * jv
  }
  acc
}

#' mpfn_is_permutation_invariant
#'
#' Part of the mpfn_native implementation; see the file header for the
#' source it follows.
#'
#' @param H See Usage.
#' @param adj See Usage.
#' @param edge_features See Usage.
#' @param perm See Usage.
#' @param T Defaults to \code{3}.
#' @param how Defaults to \code{"sum"}.
#' @param tol Defaults to \code{1e-09}.
#' @return A list with \code{invariant}, \code{max_deviation}, \code{readout}.
#' @export
mpfn_is_permutation_invariant <- function(H, adj, edge_features, perm, T = 3,
                                          how = "sum", tol = 1e-9) {
  base <- mpfn_readout(mpfn_message_passing(H, adj, edge_features, T), how)
  n <- length(H)
  inv <- integer(n)
  for (i in seq_len(n)) inv[perm[i]] <- i
  Hp <- lapply(seq_len(n), function(i) H[[inv[i]]])
  adjp <- list()
  for (v in seq_along(adj)) {
    key <- as.character(v - 1L)
    if (!is.null(adj[[key]])) {
      adjp[[as.character(perm[v - 1L])]] <- sort(sapply(adj[[key]], function(w) perm[w + 1L] - 1L))
    }
  }
  efp <- list()
  for (k in seq_along(edge_features)) {
    a <- as.integer(strsplit(names(edge_features)[k], ",")[[1]][1])
    b <- as.integer(strsplit(names(edge_features)[k], ",")[[1]][2])
    efp[[paste(perm[a + 1L] - 1L, perm[b + 1L] - 1L, sep = ",")]] <- edge_features[[k]]
  }
  other <- mpfn_readout(mpfn_message_passing(Hp, adjp, efp, T), how)
  dev <- max(abs(base - other))
  list(invariant = dev < as.numeric(tol), max_deviation = dev, readout = base)
}

#' mpfn_cheatsheet
#'
#' Part of the mpfn_native implementation; see the file header for the
#' source it follows.
#'
#' @return A character value.
#' @export
mpfn_cheatsheet <- function() {
  paste("mpfn: at least EIGHT published graph models are the same algorithm with different M_t, U_t and R. Message phase: m_v = sum_{w in N(v)} M_t(h_v, h_w, e_vw), then h_v <- U_t(h_v, m_v); readout R over the final states. The sum makes messages permutation-invariant and the READOUT MUST BE TOO, or the graph prediction changes when atoms are renumbered. Edge features carry bond type -- without them a single and a double bond between the same atoms are identical.")
}

# house entry point: the package exports one morie_<module>
morie_mpfn <- mpfn_sig
