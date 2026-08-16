% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gtrf_laplacian}
\alias{morie_gtrf_laplacian}
\title{Graph Laplacian}
\usage{
morie_gtrf_laplacian(adj, n, normalized = TRUE)
}
\arguments{
\item{adj}{Adjacency list.}
\item{n}{Number of nodes.}
\item{normalized}{Use the symmetric normalised Laplacian.}
}
\description{
\eqn{L = I - D^{-1/2} A D^{-1/2}} (normalised) or \eqn{D - A} (unnormalised).
}
\value{
Square numeric matrix.
}
\keyword{internal}
