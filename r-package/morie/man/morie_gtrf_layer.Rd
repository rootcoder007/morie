% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gtrf_layer}
\alias{morie_gtrf_layer}
\title{Graph transformer layer}
\usage{
morie_gtrf_layer(H, adj, WQ, WK, WV, W1, W2, edge_bias = NULL, norm = "batch")
}
\arguments{
\item{H}{Node feature matrix (n x d).}
\item{adj}{Adjacency list.}
\item{WQ,WK,WV}{Attention projections.}
\item{W1,W2}{Feed-forward projections.}
\item{edge_bias}{Optional per-pair scalar biases.}
\item{norm}{"batch", "layer" or "none".}
}
\description{
Attention, residual, normalisation, feed-forward, residual, normalisation, with batch normalisation by default.
}
\value{
New node feature matrix.
}
\keyword{internal}
