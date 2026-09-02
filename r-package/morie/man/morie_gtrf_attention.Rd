% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gtrf_attention}
\alias{morie_gtrf_attention}
\title{Neighbourhood-restricted attention}
\usage{
morie_gtrf_attention(H, adj, WQ, WK, WV, edge_bias = NULL) 
}
\arguments{
\item{H}{Node feature matrix (n x d).}
\item{adj}{Adjacency list keyed by node id.}
\item{WQ,WK,WV}{Projection matrices.}
\item{edge_bias}{Optional per-pair scalar biases.}
}
\description{
Standard scaled dot-product softmax attention, but only over the neighbours: dense attention would throw the graph away.
}
\value{
A list with \code{output} and \code{note}.
}
\keyword{internal}
\examples{
set.seed(1)
adj <- list(`0` = c(1, 3), `1` = c(0, 2), `2` = c(1, 3), `3` = c(0, 2))
H <- matrix(rnorm(12), 4, 3)
WQ <- matrix(rnorm(6, 0, 0.3), 2, 3)
WK <- matrix(rnorm(6, 0, 0.3), 2, 3)
WV <- matrix(rnorm(9, 0, 0.3), 3, 3)
a <- morie_gtrf_attention(H, adj, WQ, WK, WV)
dim(a$output)
}
