# SPDX-License-Identifier: AGPL-3.0-or-later
#' k-means clustering of compositions in centred log-ratio coordinates.
#'
#' Formula: minimise sum_r d_a(x_r, centre_{c(r)})^2, equivalently Euclidean k-means on clr(x_r); centres returned to the simplex by clr^-1
#'
#' @param X One composition per row; all parts strictly positive.
#' @param k Number of clusters.
#' @param init 1-based row indices of the k compositions used as starting centres; None uses the first k rows.  Supplied rather than drawn so the result is reproducible.
#' @param iters Number of Lloyd iterations to run.  A fixed count, not a tolerance, so both language arms perform identically many updates.
#'
#' @return List with ``assignment``, ``centres``, ``centres_clr``, ``inertia``, ``sizes``, ``k``, ``iters``, ``n``, ``D``.
#' @references Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The log-ratio algebra and the additive logistic normal law were taken instead from Mateu-Figueras, G., Pawlowsky-Glahn, V. and Egozcue, J. J., The normal distribution in some constrained sample spaces, arXiv:0802.2643 (published as SORT 37(1):29-56, 2013), Sects. 4.1 and 4.3, which attribute the law to Aitchison (1982, 1986); that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  The Aitchison distance is the Euclidean distance between clr coordinates, so k-means on the simplex under d_a is ordinary k-means run on clr(X); the cluster centre in clr space maps back through clr^-1 to the closed geometric mean of its members, which is the compositional centre.  Ties in the assignment step go to the LOWEST cluster index in both language arms, and an empty cluster keeps its previous centre; without both of those rules the two arms can diverge on data with exact ties.
#' @export
Compkmeans <- function(X, k, init = NULL, iters = 20L) {
  Xm <- .t1_mat(X); n <- nrow(Xm); D <- ncol(Xm)
  if (n == 0L) stop("X must have at least one composition")
  if (any(Xm <= 0)) stop("compositions must be strictly positive")
  k <- as.integer(k)
  if (k < 1L || k > n) stop("k must lie between 1 and the number of compositions")
  it <- as.integer(iters)
  if (it < 0L) stop("iters must be non-negative")
  L <- log(Xm); Zc <- L - rowMeans(L)
  seed <- if (is.null(init)) seq_len(k) else as.integer(init)
  if (length(seed) != k || anyDuplicated(seed) || any(seed < 1L | seed > n))
    stop("init must be k distinct 1-based row indices")
  Cen <- Zc[seed, , drop = FALSE]
  asg <- integer(n)
  for (step in seq_len(it)) {
    for (r in seq_len(n)) {
      dd <- rowSums(sweep(Cen, 2, Zc[r, ], "-")^2)
      asg[r] <- which.min(dd)
    }
    for (c in seq_len(k)) {
      mem <- which(asg == c)
      if (length(mem)) Cen[c, ] <- colMeans(Zc[mem, , drop = FALSE])
    }
  }
  inertia <- sum((Zc - Cen[asg, , drop = FALSE])^2)
  out <- matrix(0, k, D)
  for (c in seq_len(k)) {
    e <- exp(Cen[c, ] - max(Cen[c, ])); out[c, ] <- e / sum(e)
  }
  .t1_result(assignment = asg, centres = out, centres_clr = Cen,
             inertia = inertia, sizes = as.integer(table(factor(asg, levels = seq_len(k)))),
             k = k, iters = it, n = n, D = D,
             method = "Compositional k-means in clr coordinates")
}
