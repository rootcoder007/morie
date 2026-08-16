# Graph transformer: attention that respects the graph.
# Sources: Dwivedi, V. P. and Bresson, X. (2020) "A Generalization of
# Transformer Networks to Graphs", AAAI Workshop on Deep Learning on
# Graphs, arXiv:2012.09699 (attention restricted to the neighbourhood,
# Laplacian eigenvectors as positional encoding, batch norm in place of
# layer norm, explicit edge-feature pipeline); Vaswani, A. et al.
# (2017) "Attention Is All You Need", NIPS 2017, arXiv:1706.03762
# (the architecture being generalised); Belkin, M. and Niyogi, P.
# (2003) "Laplacian Eigenmaps for Dimensionality Reduction and Data
# Representation", Neural Computation 15(6), 1373-1396 (the
# eigenvectors).
#
# Native implementation mirroring Python morie.fn.gtrf exactly: the
# same Laplacian, the same lowest non-trivial eigenvectors as the
# positional encoding, the same random sign flip, the same
# neighbourhood-restricted attention and the same layer (residual,
# norm, feed-forward, residual, norm).

.GTRF_EPS <- 1e-12

#' Graph Laplacian
#'
#' \eqn{L = I - D^{-1/2} A D^{-1/2}} (normalised) or \eqn{D - A}
#' (unnormalised).
#'
#' @param adj Adjacency list.
#' @param n Number of nodes.
#' @param normalized Use the symmetric normalised Laplacian.
#' @return Square numeric matrix.
#' @references Belkin, M. and Niyogi, P. (2003).
#' @export
morie_gtrf_laplacian <- function(adj, n, normalized = TRUE) {
  N <- as.integer(n)
  A <- matrix(0.0, nrow = N, ncol = N)
  for (v in seq_len(N) - 1L) {
    nbrs <- as.integer(adj[[v + 1L]])
    for (w in nbrs) {
      if (v == w) next
      A[v + 1L, w + 1L] <- 1.0
      A[w + 1L, v + 1L] <- 1.0
    }
  }
  d <- rowSums(A)
  L <- matrix(0.0, nrow = N, ncol = N)
  for (i in seq_len(N)) for (j in seq_len(N)) {
    if (normalized) {
      if (d[i] <= .GTRF_EPS || d[j] <= .GTRF_EPS) {
        L[i, j] <- if (i == j) 1.0 else 0.0
      } else {
        L[i, j] <- (if (i == j) 1.0 else 0.0) -
          A[i, j] / sqrt(d[i] * d[j])
      }
    } else {
      L[i, j] <- (if (i == j) d[i] else 0.0) - A[i, j]
    }
  }
  L
}

#' Laplacian positional encoding
#'
#' The smallest non-trivial eigenvectors of the Laplacian: on a path
#' graph these are sinusoids, so the NLP positional encoding is the
#' special case.
#'
#' @param adj Adjacency list.
#' @param n Number of nodes.
#' @param dim Encoding dimension (number of eigenvectors).
#' @param normalized Use the symmetric normalised Laplacian.
#' @return A list with \code{encoding} (n x dim), \code{eigenvalues}
#'   and the sign caveat.
#' @references Dwivedi, V. P. and Bresson, X. (2020).
#' @export
morie_gtrf_lap_pe <- function(adj, n, dim = 2L, normalized = TRUE) {
  L <- morie_gtrf_laplacian(adj, n, normalized)
  ev <- eigen(L, symmetric = TRUE)
  vals <- ev$values; vecs <- ev$vectors
  order <- order(vals)
  take <- order[seq_len(as.integer(dim) + 1L)[-1L]]
  if (length(take) < as.integer(dim))
    stop(paste0("gtrf: the graph has only ", length(take),
                " non-trivial eigenvectors, ", as.integer(dim),
                " were asked for"))
  pe <- vecs[, take, drop = FALSE]
  list(encoding = pe,
       eigenvalues = vals[take],
       caveat = paste0("eigenvectors are defined up to SIGN, so the ",
                       "encoding is not unique -- the sign is flipped ",
                       "at random during training"))
}

#' Random sign flip
#'
#' Flips each eigenvector's sign with probability 1/2, per the paper.
#'
#' @param pe Positional encoding matrix (n x d).
#' @param rng Generator environment (shared with the Python arm).
#' @return Sign-flipped encoding.
#' @references Dwivedi, V. P. and Bresson, X. (2020).
#' @export
morie_gtrf_sign_flip <- function(pe, rng) {
  d <- ncol(pe)
  sgn <- ifelse(.ghc_unif(rng, d) < 0.5, 1.0, -1.0)
  pe * matrix(sgn, nrow = nrow(pe), ncol = d, byrow = TRUE)
}

#' Neighbourhood-restricted attention
#'
#' Standard scaled dot-product softmax attention, but only over the
#' neighbours: dense attention would throw the graph away.
#'
#' @param H Node feature matrix (n x d).
#' @param adj Adjacency list keyed by node id.
#' @param WQ,WK,WV Projection matrices.
#' @param edge_bias Optional per-pair scalar biases.
#' @return A list with \code{output} and \code{note}.
#' @references Dwivedi, V. P. and Bresson, X. (2020).
#' @export
morie_gtrf_attention <- function(H, adj, WQ, WK, WV, edge_bias = NULL) {
  rows <- apply(H, c(1L, 2L), as.numeric)
  dk <- ncol(WQ)
  project <- function(W, x) as.numeric(W %*% x)
  n <- nrow(rows)
  out <- matrix(0.0, nrow = n, ncol = nrow(WV))
  for (i in seq_len(n) - 1L) {
    nb <- sort(as.integer(adj[[as.character(i)]]))
    if (length(nb) == 0L) stop(paste0("gtrf: node ", i, " has no neighbours"))
    q <- project(WQ, rows[i + 1L, ])
    sc <- numeric(length(nb))
    for (jj in seq_along(nb)) {
      j <- nb[jj]
      kk <- project(WK, rows[j + 1L, ])
      s <- sum(q * kk) / sqrt(dk)
      if (!is.null(edge_bias)) {
        eb <- edge_bias[[paste0(i, ", ", j)]]
        if (is.null(eb)) eb <- edge_bias[[paste0(j, ", ", i)]]
        if (is.null(eb)) eb <- 0.0
        s <- s + as.numeric(eb)
      }
      sc[jj] <- s
    }
    m <- max(sc)
    e <- exp(sc - m)
    z <- sum(e)
    w <- e / z
    Vs <- rows[nb + 1L, , drop = FALSE] %*% t(WV)
    out[i + 1L, ] <- as.numeric(crossprod(w, Vs))
  }
  list(output = out,
       note = paste0("attention is a function of the NEIGHBOURHOOD, ",
                     "not of an arbitrary node ordering"))
}

#' Graph transformer layer
#'
#' Attention, residual, normalisation, feed-forward, residual,
#' normalisation, with batch normalisation by default.
#'
#' @param H Node feature matrix (n x d).
#' @param adj Adjacency list.
#' @param WQ,WK,WV Attention projections.
#' @param W1,W2 Feed-forward projections.
#' @param edge_bias Optional per-pair scalar biases.
#' @param norm "batch", "layer" or "none".
#' @return New node feature matrix.
#' @references Dwivedi, V. P. and Bresson, X. (2020).
#' @export
morie_gtrf_layer <- function(H, adj, WQ, WK, WV, W1, W2,
                             edge_bias = NULL, norm = "batch") {
  if (!(norm %in% c("batch", "layer", "none")))
    stop(paste0("gtrf: norm must be batch, layer or none, got ",
                deparse(norm)))
  att <- morie_gtrf_attention(H, adj, WQ, WK, WV, edge_bias)$output
  res <- H + att
  res <- .gtrf_normalize(res, norm)
  ff <- matrix(0.0, nrow = nrow(res), ncol = nrow(W2))
  for (i in seq_len(nrow(res)) - 1L) {
    h1 <- pmax(0.0, as.numeric(W1 %*% res[i + 1L, ]))
    ff[i + 1L, ] <- as.numeric(W2 %*% h1)
  }
  out <- res + ff
  .gtrf_normalize(out, norm)
}

#' .gtrf_normalize
#'
#' A step of the gtrf_native implementation. Called by \code{morie_gtrf_layer}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param X A matrix; indexed by row and column.
#' @param how One of \code{"batch"}, \code{"none"}.
#' @return One of two values, depending on the branch taken.
#' @export
.gtrf_normalize <- function(X, how) {
  if (how == "none") return(X)
  n <- nrow(X); d <- ncol(X)
  if (how == "batch") {
    mu <- colSums(X) / n
    sd <- sqrt(colSums((X - matrix(mu, nrow = n, ncol = d, byrow = TRUE))^2) /
                 n + 1e-5)
    sweep(X, 2L, mu) / matrix(sd, nrow = n, ncol = d, byrow = TRUE)
  } else {
    out <- matrix(0.0, nrow = n, ncol = d)
    for (i in seq_len(n) - 1L) {
      mu <- mean(X[i + 1L, ])
      sd <- sqrt(sum((X[i + 1L, ] - mu)^2) / d + 1e-5)
      out[i + 1L, ] <- (X[i + 1L, ] - mu) / sd
    }
    out
  }
}

# house entry point: the package exports one morie_<module>
morie_gtrf <- morie_gtrf_laplacian
