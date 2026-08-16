# GraphSAGE: embeddings for nodes the model has never seen.
# Sources: Hamilton, W. L., Ying, R. and Leskovec, J. (2017) "Inductive
# Representation Learning on Large Graphs", NeurIPS 2017,
# arXiv:1706.02216 (Algorithm 1, mean / max-pooling / LSTM
# aggregators, fixed-size neighbour sampling); Kipf, T. N. and
# Welling, M. (2017) "Semi-Supervised Classification with Graph
# Convolutional Networks", ICLR 2017, arXiv:1609.02907 (the
# transductive GCN GraphSAGE extends to the inductive setting).
#
# Native implementation mirroring Python morie.fn.gsageemd exactly:
# the same three aggregators (mean, max_pool, lstm_order), the same
# concatenation-then-linear transform with optional normalisation,
# the same fixed-size neighbour sampling, and the same unsupervised
# loss of Sec. 3.2. The shared generator is used for sampling and the
# random permutations so both arms produce the same stream.

.GSAGE_EPS <- 1e-12
.GSAGE_AGGS <- c("mean", "max_pool", "lstm_order")

#' Permutation-invariant neighbour aggregation
#'
#' The three aggregators of Hamilton et al. 2017: \code{mean} (the
#' default, nearly the transductive GCN's rule),
#' \code{max_pool} (pass each neighbour through a linear layer and
#' take an element-wise max), and \code{lstm_order} (returns the
#' vectors in the order given -- the paper's point that an LSTM
#' aggregator is not symmetric and must be fed a random permutation).
#'
#' @param vectors A list of numeric vectors.
#' @param how One of \code{"mean"}, \code{"max_pool"}, \code{"lstm_order"}.
#' @param W Optional weight matrix used by \code{max_pool}.
#' @return A single numeric vector.
#' @references Hamilton, W. L. et al. (2017).
#' @export
.gsage_rows <- function(x) {
  # k.mat's contract: accept a list of vectors OR a matrix, yield rows.
  if (is.matrix(x) || is.data.frame(x)) {
    m <- as.matrix(x)
    storage.mode(m) <- "double"
    return(lapply(seq_len(nrow(m)), function(i) as.numeric(m[i, ])))
  }
  lapply(x, as.numeric)
}

#' morie_gsageemd_aggregate
#'
#' Part of the gsageemd_native implementation; see the file header for
#' the source it follows.
#'
#' @param vectors See Usage.
#' @param how Defaults to \code{"mean"}.
#' @param W Defaults to \code{NULL}.
#' @return One of two values, depending on the branch taken.
#' @export
morie_gsageemd_aggregate <- function(vectors, how = "mean", W = NULL) {
  if (!(how %in% .GSAGE_AGGS))
    stop(paste0("gsageemd: aggregator must be one of ",
                paste(.GSAGE_AGGS, collapse = ", "), ", got ",
                deparse(how)))
  V <- .gsage_rows(vectors)
  if (length(V) == 0L) stop("gsageemd: no neighbours to aggregate")
  d <- length(V[[1L]])
  if (how == "mean") {
    out <- rep(0.0, d)
    for (i in seq_along(V)) for (f in seq_len(d)) out[f] <- out[f] + V[[i]][f]
    out / length(V)
  } else if (how == "max_pool") {
    if (is.null(W)) {
      out <- rep(-Inf, d)
      for (i in seq_along(V)) for (f in seq_len(d))
        if (V[[i]][f] > out[f]) out[f] <- V[[i]][f]
      out
    } else {
      H <- matrix(0, nrow = length(V), ncol = nrow(W))
      for (i in seq_along(V)) {
        for (o in seq_len(nrow(W))) {
          s <- 0.0
          for (j in seq_len(d)) s <- s + W[o, j] * V[[i]][j]
          H[i, o] <- max(0.0, s)
        }
      }
      apply(H, 2L, max)
    }
  } else {
    V[[1L]]
  }
}

#' Fixed-size neighbour sample
#'
#' Samples with replacement when the neighbourhood is smaller than the
#' budget; the budget is what bounds the per-batch cost regardless of
#' node degree.
#'
#' @param adj Adjacency list keyed by node.
#' @param v Node whose neighbours are sampled.
#' @param size Sample size.
#' @param rng Generator environment (shared with the Python arm).
#' @return Integer vector of neighbour ids.
#' @references Hamilton, W. L. et al. (2017).
#' @export
morie_gsageemd_sample <- function(adj, v, size, rng) {
  nb <- sort(as.integer(adj[[as.character(v)]]))
  if (length(nb) == 0L)
    stop(paste0("gsageemd: node ", v, " has no neighbours"))
  s <- as.integer(size)
  if (s < 1L) stop("gsageemd: the sample size must be at least 1")
  vapply(seq_len(s), function(.)
    nb[(floor(.ghc_unif(rng, 1L) * length(nb)) %% length(nb)) + 1L],
    integer(1))
}

.gsage_norm <- function(v) {
  n <- sqrt(sum(v * v))
  if (n <= .GSAGE_EPS) v else v / n
}

#' One GraphSAGE layer
#'
#' Algorithm 1 of Hamilton et al. 2017 for a single depth: aggregate
#' the neighbourhood, concatenate with the node's own previous
#' representation, linear transform with ReLU, optionally L2-normalise.
#'
#' @param H Node feature matrix (n x d).
#' @param adj Adjacency list keyed by node id.
#' @param W Linear transform (n_out x (d + d_neigh)).
#' @param how Aggregator.
#' @param sizes Optional fixed sample size per node.
#' @param rng Generator environment.
#' @param normalize L2-normalise the output.
#' @return Matrix of new node representations.
#' @references Hamilton, W. L. et al. (2017).
#' @export
morie_gsageemd_layer <- function(H, adj, W, how = "mean", sizes = NULL,
                                 rng = NULL, normalize = TRUE) {
  n <- nrow(H)
  out <- matrix(0.0, nrow = n, ncol = nrow(W))
  for (v in seq_len(n) - 1L) {
    if (is.null(sizes)) {
      nb <- sort(as.integer(adj[[as.character(v)]]))
    } else {
      nb <- morie_gsageemd_sample(adj, v, sizes, rng)
    }
    if (length(nb) == 0L)
      stop(paste0("gsageemd: node ", v, " has no neighbours"))
    agg <- morie_gsageemd_aggregate(lapply(nb + 1L, function(u) H[u, ]),
                                    how = how, W = W)
    cat <- c(H[v + 1L, ], agg)
    if (ncol(W) != length(cat))
      stop(paste0("gsageemd: W expects ", ncol(W), " inputs but the ",
                  "concatenation is ", length(cat)))
    z <- pmax(0.0, as.numeric(W %*% cat))
    out[v + 1L, ] <- if (normalize) .gsage_norm(z) else z
  }
  out
}

#' K-hop GraphSAGE embedding
#'
#' Applies K layers, so K hops. Parameters are shared across nodes,
#' which is what lets an unseen node be embedded by a forward pass
#' rather than by retraining.
#'
#' @param features Node feature matrix (n x d).
#' @param adj Adjacency list.
#' @param Ws List of weight matrices, one per layer.
#' @param how Aggregator.
#' @param sizes Optional fixed sample size per layer.
#' @param seed Integer seed for the shared generator.
#' @return A list with \code{embeddings}, \code{depth},
#'   \code{aggregator}, \code{per_batch_bound}, \code{method} and
#'   \code{note}.
#' @references Hamilton, W. L. et al. (2017).
#' @export
morie_gsageemd_embed <- function(features, adj, Ws, how = "mean",
                                 sizes = NULL, seed = 0) {
  e <- .ghc_rng(as.integer(seed))
  H <- apply(features, c(1L, 2L), as.numeric)
  for (W in Ws) {
    H <- morie_gsageemd_layer(H, adj, W, how = how, sizes = sizes, rng = e,
                              normalize = TRUE)
  }
  list(estimate = H, embeddings = H, depth = length(Ws),
       aggregator = how,
       per_batch_bound = if (is.null(sizes)) NULL else
         as.integer(sizes) ^ length(Ws),
       method = "GraphSAGE; Hamilton, Ying & Leskovec (2017) Algorithm 1",
       note = paste0("parameters are shared across nodes, so an ",
                     "unseen node is embedded by a forward pass -- ",
                     "inductive, not transductive"))
}

#' Unsupervised graph-based loss
#'
#' Sec. 3.2 of Hamilton et al. 2017: nearby nodes agree, sampled
#' negatives disagree.
#'
#' @param z_u Anchor embedding.
#' @param z_v Positive (neighbour) embedding.
#' @param z_negatives List of negative embeddings.
#' @return Scalar loss.
#' @references Hamilton, W. L. et al. (2017).
#' @export
morie_gsageemd_loss <- function(z_u, z_v, z_negatives) {
  dot <- function(a, b) sum(a * b)
  pos <- log(max(1.0 / (1.0 + exp(-dot(z_u, z_v))), .GSAGE_EPS))
  neg <- sum(vapply(z_negatives, function(zn)
    log(max(1.0 / (1.0 + exp(dot(z_u, zn))), .GSAGE_EPS)),
    numeric(1)))
  -(pos + neg)
}

# house entry point: the package exports one morie_<module>
morie_gsageemd <- morie_gsageemd_aggregate
