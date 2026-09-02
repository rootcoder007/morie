% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_grphmr_spatial}
\alias{morie_grphmr_spatial}
\title{Spatial bias from the shortest-path distance matrix}
\usage{
morie_grphmr_spatial(distance, b_table, unreachable_bias = -10) 
}
\arguments{
\item{distance}{N x N matrix from \code{shortest_path_matrix}.}
\item{b_table}{Numeric vector of learnable biases indexed by   distance.}
\item{unreachable_bias}{Bias for disconnected pairs; defaults to   \code{-10}.}
}
\description{
Looks up \code{b_table[d]} (clipped to the table size) and uses \code{unreachable_bias} for disconnected pairs.
}
\value{
A list with \code{bias}, \code{unreachable_bias},   \code{note}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
M <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
morie_grphmr_spatial(M, V)
}
