% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gsageemd_sample}
\alias{morie_gsageemd_sample}
\title{Fixed-size neighbour sample}
\usage{
morie_gsageemd_sample(adj, v, size, rng)
}
\arguments{
\item{adj}{Adjacency list keyed by node.}
\item{v}{Node whose neighbours are sampled.}
\item{size}{Sample size.}
\item{rng}{Generator environment (shared with the Python arm).}
}
\description{
Samples with replacement when the neighbourhood is smaller than the budget; the budget is what bounds the per-batch cost regardless of node degree.
}
\value{
Integer vector of neighbour ids.
}
\keyword{internal}
