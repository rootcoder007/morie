% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gsageemd_embed}
\alias{morie_gsageemd_embed}
\title{K-hop GraphSAGE embedding}
\usage{
morie_gsageemd_embed(features, adj, Ws, how = "mean", sizes = NULL, seed = 0)
}
\arguments{
\item{features}{Node feature matrix (n x d).}
\item{adj}{Adjacency list.}
\item{Ws}{List of weight matrices, one per layer.}
\item{how}{Aggregator.}
\item{sizes}{Optional fixed sample size per layer.}
\item{seed}{Integer seed for the shared generator.}
}
\description{
Applies K layers, so K hops. Parameters are shared across nodes, which is what lets an unseen node be embedded by a forward pass rather than by retraining.
}
\value{
A list with \code{embeddings}, \code{depth},   \code{aggregator}, \code{per_batch_bound}, \code{method} and   \code{note}.
}
\keyword{internal}
\examples{
adj <- list(`0` = c(`1` = 1, `3` = 1), `1` = c(`0` = 1, `2` = 1),
            `2` = c(`1` = 1, `3` = 1), `3` = c(`2` = 1, `0` = 1))
feats <- matrix(runif(12), 4, 3)
W1 <- matrix(rnorm(6 * 4, 0, 0.3), 4, 6)
W2 <- matrix(rnorm(8 * 2, 0, 0.3), 2, 8)
emb <- morie_gsageemd_embed(feats, adj, list(W1, W2), how = "mean", seed = 1)
dim(emb)
}
