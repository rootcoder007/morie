% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_grphmr}
\alias{morie_grphmr}
\title{Graphormer attention with structural encodings}
\usage{
morie_grphmr(H, WQ, WK, WV, bias, edge_bias = NULL)
}
\arguments{
\item{H}{Node features, an N x d matrix.}
\item{WQ,WK,WV}{Projection matrices, each d x d.}
\item{bias}{Spatial bias matrix, N x N.}
\item{edge_bias}{Optional edge-feature bias as a length-N^2 vector   indexed row-major, or a named numeric vector keyed by   \code{"i,j"}.}
}
\description{
A standard scaled-dot-product attention layer, augmented with the three structural encodings of Ying et al. (2021): a per-degree centrality vector added to the node features; a per-shortest-path distance bias added inside the softmax; and an edge-feature contribution along the same paths.
}
\value{
A list with \code{estimate}, \code{output}, \code{weights},   \code{method}, \code{note}.
}
\keyword{internal}
