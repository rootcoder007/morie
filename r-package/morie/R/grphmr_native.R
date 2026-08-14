# Graphormer attention: a standard Transformer with structural encodings.
# Sources: Ying, C., Cai, T., Luo, S., Zheng, S., Ke, G., He, D., Shen,
# Y. & Liu, T.-Y. (2021) Do Transformers Really Perform Bad for Graph
# Representation?, NeurIPS 2021, 28877-28888, arXiv:2106.05234 -- the
# three encodings (centrality by degree, spatial bias by shortest-path
# distance, edge features along the path); Vaswani, A. et al. (2017)
# Attention Is All You Need, NIPS 2017, 5998-6008 -- the underlying
# scaled-dot-product attention.
#
# Native R port mirroring morie.fn.grphmr exactly. The Python arm uses
# python-native dicts and a 1-based BFS over adjacency lists; we keep
# the same graph convention (vertex keys 0..N-1 stored as the integer
# indices of the adjacency list vectors) and reproduce the same bias
# values, the same unreachable default, and the same attention logits.

#' Graphormer attention with structural encodings
#'
#' A standard scaled-dot-product attention layer, augmented with the
#' three structural encodings of Ying et al. (2021): a per-degree
#' centrality vector added to the node features; a per-shortest-path
#' distance bias added inside the softmax; and an edge-feature
#' contribution along the same paths.
#'
#' @param H Node features, an N x d matrix.
#' @param WQ,WK,WV Projection matrices, each d x d.
#' @param bias Spatial bias matrix, N x N.
#' @param edge_bias Optional edge-feature bias as a length-N^2 vector
#'   indexed row-major, or a named numeric vector keyed by
#'   \code{"i,j"}.
#' @return A list with \code{estimate}, \code{output}, \code{weights},
#'   \code{method}, \code{note}.
#' @references Ying, C. et al. (2021). Do Transformers Really Perform
#'   Bad for Graph Representation? NeurIPS 2021, 28877-28888.
#' @export
morie_grphmr <- function(H, WQ, WK, WV, bias, edge_bias = NULL) {
  X <- as.matrix(H)
  storage.mode(X) <- "double"
  n <- nrow(X); dk <- ncol(X)
  bias <- as.matrix(bias); storage.mode(bias) <- "double"
  if (nrow(bias) != n || ncol(bias) != n)
    stop("grphmr: bias must be N x N")

  WQ <- as.matrix(WQ); storage.mode(WQ) <- "double"
  WK <- as.matrix(WK); storage.mode(WK) <- "double"
  WV <- as.matrix(WV); storage.mode(WV) <- "double"
  if (nrow(WQ) != dk || ncol(WQ) != dk) stop("grphmr: WQ shape")
  if (nrow(WK) != dk || ncol(WK) != dk) stop("grphmr: WK shape")
  if (nrow(WV) != dk || ncol(WV) != dk) stop("grphmr: WV shape")

  eb <- NULL
  if (!is.null(edge_bias)) {
    eb <- matrix(0, n, n)
    if (is.matrix(edge_bias) || is.numeric(edge_bias)) {
      M <- as.matrix(edge_bias); storage.mode(M) <- "double"
      if (nrow(M) != n || ncol(M) != n) stop("grphmr: edge_bias shape")
      eb <- M
    } else {
      for (k in seq_along(edge_bias)) {
        key <- names(edge_bias)[k]
        ij <- strsplit(key, ",", fixed = TRUE)[[1]]
        i <- as.integer(ij[1]); j <- as.integer(ij[2])
        eb[i + 1L, j + 1L] <- as.numeric(edge_bias[k])
      }
    }
  }

  Q <- X %*% WQ
  K <- X %*% WK
  V <- X %*% WV
  sc <- (Q %*% t(K)) / sqrt(dk) + bias
  if (!is.null(eb)) sc <- sc + eb
  out <- matrix(0, n, dk)
  weights <- matrix(0, n, n)
  for (i in seq_len(n)) {
    row <- sc[i, ]
    m <- max(row)
    e <- exp(row - m)
    z <- sum(e)
    w <- e / z
    weights[i, ] <- w
    out[i, ] <- as.numeric(t(V) %*% w)
  }
  list(estimate = out, output = out, weights = weights,
       method = paste0("Graphormer attention with centrality, spatial ",
                       "and edge encodings; Ying et al. (2021)"),
       note = paste0("the architecture is a STANDARD Transformer; the ",
                     "structural encodings are what was missing"))
}

#' Graphormer centrality encoding
#'
#' Adds a learnable vector indexed by node degree to the input
#' features. Attention is computed from features, so without this a hub
#' and a leaf with identical features are indistinguishable.
#'
#' @param adj Adjacency list: list of numeric neighbour vectors.
#' @param n Number of nodes.
#' @param z_in Length-(maxdeg+1) list of length-d vectors.
#' @param z_out Optional separate table for out-degree (directed).
#' @param directed Logical; if TRUE uses in/out degree separately.
#' @return A list with \code{encoding}, \code{degrees},
#'   \code{note}.
#' @export
morie_grphmr_centrality <- function(adj, n, z_in, z_out = NULL,
                                    directed = FALSE) {
  N <- as.integer(n)
  deg_in <- integer(N)
  deg_out <- integer(N)
  for (v in seq_len(N) - 1L) {
    nbrs <- unique(adj[[v + 1L]])
    nbrs <- nbrs[nbrs != v]
    deg_out[v + 1L] <- length(nbrs)
  }
  for (w in seq_len(N) - 1L) {
    src <- which(vapply(seq_len(N), function(v) w %in% adj[[v]], logical(1)))
    deg_in[w + 1L] <- length(src) - 1L
  }
  if (!directed) {
    deg_in <- vapply(seq_len(N) - 1L, function(v)
      length(setdiff(unique(adj[[v + 1L]]), v)), integer(1))
    deg_out <- deg_in
  }
  out <- vector("list", N)
  for (v in seq_len(N)) {
    d <- min(deg_in[v], length(z_in) - 1L)
    vec <- as.numeric(z_in[[d + 1L]])
    if (directed && !is.null(z_out)) {
      o <- min(deg_out[v], length(z_out) - 1L)
      vec <- vec + as.numeric(z_out[[o + 1L]])
    }
    out[[v]] <- vec
  }
  list(encoding = out, degrees = deg_in,
       note = "indexed by degree, added at the INPUT layer")
}

#' All-pairs shortest path distances
#'
#' BFS from every node; pairs with no path are assigned
#' \code{UNREACHABLE} so they can take a special bias rather than
#' producing NaNs downstream.
#'
#' @param adj Adjacency list: list of numeric neighbour vectors.
#' @param n Number of nodes.
#' @return A list with \code{distance} (N x N integer matrix),
#'   \code{unreachable}, \code{n_unreachable}.
#' @export
morie_grphmr_sp <- function(adj, n) {
  N <- as.integer(n)
  D <- matrix(-1L, N, N)
  for (s in seq_len(N) - 1L) {
    D[s + 1L, s + 1L] <- 0L
    seen <- rep(FALSE, N); seen[s + 1L] <- TRUE
    frontier <- s
    d <- 0L
    while (length(frontier) > 0L) {
      d <- d + 1L
      nxt <- integer(0)
      for (v in frontier) {
        nbrs <- setdiff(unique(adj[[v + 1L]]), v)
        for (w in nbrs) {
          if (!seen[w + 1L]) {
            seen[w + 1L] <- TRUE
            D[s + 1L, w + 1L] <- d
            nxt <- c(nxt, w)
          }
        }
      }
      frontier <- nxt
    }
  }
  list(distance = D, unreachable = -1L,
       n_unreachable = sum(D == -1L))
}

#' Spatial bias from the shortest-path distance matrix
#'
#' Looks up \code{b_table[d]} (clipped to the table size) and uses
#' \code{unreachable_bias} for disconnected pairs.
#'
#' @param distance N x N matrix from \code{shortest_path_matrix}.
#' @param b_table Numeric vector of learnable biases indexed by
#'   distance.
#' @param unreachable_bias Bias for disconnected pairs; defaults to
#'   \code{-10}.
#' @return A list with \code{bias}, \code{unreachable_bias},
#'   \code{note}.
#' @export
morie_grphmr_spatial <- function(distance, b_table, unreachable_bias = -10) {
  D <- matrix(as.integer(distance), nrow = nrow(distance))
  N <- nrow(D)
  ub <- as.numeric(unreachable_bias)
  out <- matrix(0, N, N)
  for (i in seq_len(N)) {
    for (j in seq_len(N)) {
      if (D[i, j] == -1L) out[i, j] <- ub
      else {
        d <- min(D[i, j], length(b_table) - 1L)
        out[i, j] <- as.numeric(b_table[d + 1L])
      }
    }
  }
  list(bias = out, unreachable_bias = ub,
       note = paste0("a bias inside the softmax keeps distant nodes ",
                     "reachable but discouraged"))
}

#' Edge-feature bias averaged along the shortest path
#'
#' For each pair the edge features along the path are weighted by
#' \code{w_table[step]} and averaged across the path length. Bond
#' type is a property of neither endpoint, so it has to enter the
#' model here.
#'
#' @param paths Named list of integer vectors, one per (i, j) pair.
#' @param edge_features Named list of numeric vectors keyed by edge
#'   \code{"u,v"} (and \code{"v,u"}).
#' @param w_table Length-K numeric vector of step weights.
#' @return A list with \code{edge_bias} (named numeric vector),
#'   \code{note}.
#' @export
morie_grphmr_edge <- function(paths, edge_features, w_table) {
  out <- list()
  for (key in names(paths)) {
    path <- paths[[key]]
    if (length(path) == 0L) { out[[key]] <- 0.0; next }
    acc <- 0.0
    for (step in seq_along(path)) {
      e <- path[step]
      ek <- paste0(e[1], ",", e[2])
      rk <- paste0(e[2], ",", e[1])
      f <- if (!is.null(edge_features[[ek]])) edge_features[[ek]]
           else if (!is.null(edge_features[[rk]])) edge_features[[rk]]
           else stop("grphmr: no features for edge ", ek)
      fv <- as.numeric(f)
      s <- min(step, length(w_table))
      wv <- as.numeric(w_table[[s]])
      L <- min(length(fv), length(wv))
      acc <- acc + sum(fv[seq_len(L)] * wv[seq_len(L)])
    }
    out[[key]] <- acc / length(path)
  }
  list(edge_bias = out,
       note = paste0("edge information cannot reach the model through ",
                     "node features"))
}
