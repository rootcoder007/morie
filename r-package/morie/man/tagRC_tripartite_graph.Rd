% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{tagRC_tripartite_graph}
\alias{tagRC_tripartite_graph}
\title{tagRC_tripartite_graph}
\usage{
tagRC_tripartite_graph(triples) 
}
\arguments{
\item{triples}{See Usage.}
}
\description{
Part of the tagRC_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{adjacency}, \code{nodes}, \code{n_nodes}, \code{n_triples}, \code{note}.
}
\keyword{internal}
\examples{
D <- data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9))
tagRC_tripartite_graph(D)
}
