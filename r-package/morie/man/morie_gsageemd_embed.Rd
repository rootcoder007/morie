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
