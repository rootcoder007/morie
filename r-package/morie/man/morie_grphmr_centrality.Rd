% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_grphmr_centrality}
\alias{morie_grphmr_centrality}
\title{Graphormer centrality encoding}
\usage{
morie_grphmr_centrality(adj, n, z_in, z_out = NULL, directed = FALSE)
}
\arguments{
\item{adj}{Adjacency list: list of numeric neighbour vectors.}
\item{n}{Number of nodes.}
\item{z_in}{Length-(maxdeg+1) list of length-d vectors.}
\item{z_out}{Optional separate table for out-degree (directed).}
\item{directed}{Logical; if TRUE uses in/out degree separately.}
}
\description{
Adds a learnable vector indexed by node degree to the input features. Attention is computed from features, so without this a hub and a leaf with identical features are indistinguishable.
}
\value{
A list with \code{encoding}, \code{degrees},   \code{note}.
}
\keyword{internal}
\examples{
morie_grphmr_centrality(adj = c(1, 2, 3, 4, 5, 6, 7, 8), n = 5L, z_in = c(1, 2, 3, 4, 5, 6, 7, 8))
}
