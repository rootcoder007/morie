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
\examples{
adj <- list(`0` = c(1, 3), `1` = c(0, 2), `2` = c(1, 3), `3` = c(0, 2))
L <- morie_gtrf_laplacian(adj, n = 4)
c(nrow(L), all(abs(diag(L) - 1) < 1e-9))
}
