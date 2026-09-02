% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_grphmr_sp}
\alias{morie_grphmr_sp}
\title{All-pairs shortest path distances}
\usage{
morie_grphmr_sp(adj, n) 
}
\arguments{
\item{adj}{Adjacency list: list of numeric neighbour vectors.}
\item{n}{Number of nodes.}
}
\description{
BFS from every node; pairs with no path are assigned \code{UNREACHABLE} so they can take a special bias rather than producing NaNs downstream.
}
\value{
A list with \code{distance} (N x N integer matrix),   \code{unreachable}, \code{n_unreachable}.
}
\keyword{internal}
