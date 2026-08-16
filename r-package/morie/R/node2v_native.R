# Sources: Grover, A. & Leskovec, J. (2016) "node2vec: Scalable Feature
# Learning for Networks", KDD '16, 855-864, doi:10.1145/2939672.2939754,
# arXiv:1607.00653 (Sec. 2; Sec. 3 eq. (1); Sec. 3.2.2 second-order walk
# with alpha_pq keyed on d_tx); Mikolov, T., Sutskever, I., Chen, K.,
# Corrado, G. & Dean, J. (2013) "Distributed Representations of Words
# and Phrases and their Compositionality", NIPS 2013, 3111-3119,
# arXiv:1310.4546 (skip-gram with negative sampling); Perozzi, B.,
# Al-Rfou, R. & Skiena, S. (2014) "DeepWalk: Online Learning of Social
# Representations", KDD '14, 701-710, doi:10.1145/2623330.2623732
# (the uniform random walk node2vec generalises).
#
# Native implementation mirroring Python morie.fn.node2v exactly: the
# same alpha_pq tabulated from d_tx in {0,1,2}, the same first-order
# step (uniform/weight-proportional) when the previous node is NULL,
# the same inverse-CDF draw on the normalised transition probabilities,
# the same num_walks walks from every node, and the same skip-gram
# pairs within the window.

#' node2v_alpha_pq
#'
#' Part of the node2v_native implementation; see the file header for the
#' source it follows.
#'
#' @param d_tx See Usage.
#' @param p See Usage.
#' @param q See Usage.
#' @return Nothing; this branch always raises.
#' @export
node2v_alpha_pq <- function(d_tx, p, q) {
  d <- as.integer(d_tx)
  pp <- as.numeric(p)
  qq <- as.numeric(q)
  if (!is.finite(pp) || pp <= 0 || !is.finite(qq) || qq <= 0)
    stop("node2v: p and q must be positive")
  if (d == 0L) return(1 / pp)
  if (d == 1L) return(1)
  if (d == 2L) return(1 / qq)
  stop("node2v: d_tx must be 0, 1 or 2 for a second-order walk, got ",
       d)
}

#' node2v_dist
#'
#' Part of the node2v_native implementation; see the file header for the
#' source it follows.
#'
#' @param adj See Usage.
#' @param t See Usage.
#' @param x See Usage.
#' @return A numeric value.
#' @export
node2v_dist <- function(adj, t, x) {
  if (isTRUE(t == x)) return(0L)
  nb_t <- adj[[as.character(t)]]
  if (is.null(nb_t)) nb_t <- adj[[t]]
  if (!is.null(nb_t) && x %in% nb_t) return(1L)
  2L
}

#' node2v_transition_probabilities
#'
#' Part of the node2v_native implementation; see the file header for the
#' source it follows.
#'
#' @param adj See Usage.
#' @param t See Usage.
#' @param v See Usage.
#' @param p See Usage.
#' @param q See Usage.
#' @param weights Defaults to \code{NULL}.
#' @return A list with \code{nodes}, \code{probabilities}, \code{unnormalized}, \code{Z}.
#' @export
node2v_transition_probabilities <- function(adj, t, v, p, q,
                                            weights = NULL) {
  v_key <- v
  nb <- adj[[v_key]]
  if (is.null(nb)) nb <- adj[[as.character(v)]]
  if (is.null(nb)) nb <- character(0)
  nb <- sort(unique(nb))
  if (length(nb) == 0L)
    stop("node2v: node ", deparse(v), " has no neighbours")
  pi <- numeric(length(nb))
  for (i in seq_along(nb)) {
    x <- nb[i]
    w <- 1
    if (!is.null(weights)) {
      key <- paste0(as.character(v), "\r", as.character(x))
      wkey1 <- paste0(as.character(v), "|", as.character(x))
      if (!is.null(weights[[key]])) w <- as.numeric(weights[[key]])
      else if (!is.null(weights[[wkey1]])) w <- as.numeric(weights[[wkey1]])
      else if (!is.null(weights[[paste0(as.character(v), ",", as.character(x))]]))
        w <- as.numeric(weights[[paste0(as.character(v), ",", as.character(x))]])
    }
    if (is.null(t)) {
      a <- 1
    } else {
      a <- node2v_alpha_pq(node2v_dist(adj, t, x), p, q)
    }
    pi[i] <- a * w
  }
  Z <- sum(pi)
  list(nodes = nb,
       probabilities = pi / Z,
       unnormalized = pi, Z = Z)
}

#' node2v_walk
#'
#' Part of the node2v_native implementation; see the file header for the
#' source it follows.
#'
#' @param adj See Usage.
#' @param start See Usage.
#' @param length See Usage.
#' @param p Defaults to \code{1}.
#' @param q Defaults to \code{1}.
#' @param rng Defaults to \code{NULL}.
#' @param weights Defaults to \code{NULL}.
#' @return The value of \code{path}, as built in the body.
#' @export
node2v_walk <- function(adj, start, length, p = 1, q = 1, rng = NULL,
                        weights = NULL) {
  if (is.null(rng)) {
    rng <- .ghc_rng(0)
    own <- TRUE
  } else {
    own <- FALSE
  }
  path <- c(start)
  prev <- NULL
  for (step in seq_len(as.integer(length) - 1L)) {
    tp <- node2v_transition_probabilities(adj, prev, path[length(path)],
                                          p, q, weights)
    u <- .ghc_unif(rng, 1L)
    acc <- 0
    nxt <- tp$nodes[length(tp$nodes)]
    for (i in seq_along(tp$nodes)) {
      acc <- acc + tp$probabilities[i]
      if (u <= acc) { nxt <- tp$nodes[i]; break }
    }
    prev <- path[length(path)]
    path <- c(path, nxt)
  }
  path
}

#' morie_node2v
#'
#' Part of the node2v_native implementation; see the file header for the
#' source it follows.
#'
#' @param adj See Usage.
#' @param num_walks Defaults to \code{10}.
#' @param length Defaults to \code{10}.
#' @param p Defaults to \code{1}.
#' @param q Defaults to \code{1}.
#' @param seed Defaults to \code{0}.
#' @param weights Defaults to \code{NULL}.
#' @return A list with \code{estimate}, \code{walks}, \code{p}, \code{q}, \code{n_walks}, \code{length}, \code{method}, \code{note}.
#' @export
morie_node2v <- function(adj, num_walks = 10, length = 10, p = 1, q = 1,
                         seed = 0, weights = NULL) {
  rng <- .ghc_rng(as.numeric(seed))
  out <- list()
  for (w in seq_len(as.integer(num_walks))) {
    nodes <- names(adj)
    if (is.null(nodes)) {
      if (is.list(adj)) {
        nm <- vapply(adj, function(e) is.character(e) || is.numeric(e),
                     logical(1))
        nodes <- if (any(nm)) names(adj)[nm] else names(adj)
      } else {
        nodes <- names(adj)
      }
    }
    if (is.null(nodes)) nodes <- as.character(seq_along(adj))
    for (v in nodes) {
      out[[length(out) + 1L]] <-
        node2v_walk(adj, v, length, p, q, rng, weights)
    }
  }
  list(estimate = out, walks = out, p = as.numeric(p), q = as.numeric(q),
       n_walks = length(out), length = as.integer(length),
       method = "second-order biased random walk; Grover & Leskovec (2016) Sec. 3.2.2",
       note = "large q keeps the walk local (BFS-like), small q pushes it outward (DFS-like); p prices returning")
}

node2v_generate_walks <- morie_node2v

#' node2v_skipgram_pairs
#'
#' Part of the node2v_native implementation; see the file header for the
#' source it follows.
#'
#' @param walks See Usage.
#' @param window Defaults to \code{2}.
#' @return The value of \code{pairs}, as built in the body.
#' @export
node2v_skipgram_pairs <- function(walks, window = 2) {
  w <- as.integer(window)
  if (w < 1L)
    stop("node2v: the window must be at least 1")
  pairs <- list()
  for (path in walks) {
    n <- length(path)
    for (i in seq_len(n)) {
      lo <- max(1L, i - w)
      hi <- min(n, i + w)
      for (j in lo:hi) {
        if (j != i) pairs[[length(pairs) + 1L]] <- c(path[i], path[j])
      }
    }
  }
  pairs
}

#' node2v_cheatsheet
#'
#' Part of the node2v_native implementation; see the file header for the
#' source it follows.
#'
#' @return A character value.
#' @export
node2v_cheatsheet <- function() {
  paste("node2v: graph as document, walk as sentence, skip-gram on ",
        "top. The point is that NO sampling strategy wins everywhere: ",
        "BFS gives a low-variance local structural view, DFS a ",
        "macroscopic community view, and real networks mix both. A ",
        "SECOND-ORDER walk interpolates -- having come from t, the ",
        "bias to x is 1/p if returning, 1 if x neighbours t, 1/q ",
        "otherwise. Large q stays local, small q roams. A first-order ",
        "walk cannot express this.")
}

node2vec <- morie_node2v
