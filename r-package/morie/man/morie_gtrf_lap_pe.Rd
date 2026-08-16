% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gtrf_lap_pe}
\alias{morie_gtrf_lap_pe}
\title{Laplacian positional encoding}
\usage{
morie_gtrf_lap_pe(adj, n, dim = 2L, normalized = TRUE)
}
\arguments{
\item{adj}{Adjacency list.}
\item{n}{Number of nodes.}
\item{dim}{Encoding dimension (number of eigenvectors).}
\item{normalized}{Use the symmetric normalised Laplacian.}
}
\description{
The smallest non-trivial eigenvectors of the Laplacian: on a path graph these are sinusoids, so the NLP positional encoding is the special case.
}
\value{
A list with \code{encoding} (n x dim), \code{eigenvalues}   and the sign caveat.
}
\keyword{internal}
