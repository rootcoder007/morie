% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{node2v_walk}
\alias{node2v_walk}
\title{node2v_walk}
\usage{
node2v_walk(adj, start, length, p = 1, q = 1, rng = NULL, weights = NULL)
}
\arguments{
\item{adj}{Passed to \code{node2v_transition_probabilities}.}
\item{start}{Passed to \code{c}.}
\item{length}{Coerced to integer by the body, with \code{as.integer}.}
\item{p}{Passed to \code{node2v_transition_probabilities}. Defaults to \code{1}.}
\item{q}{Passed to \code{node2v_transition_probabilities}. Defaults to \code{1}.}
\item{rng}{Optional; may be \code{NULL}. Passed to \code{.ghc_unif}.}
\item{weights}{Passed to \code{node2v_transition_probabilities}.}
}
\description{
Part of the node2v_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{path}, as built in the body.
}
\keyword{internal}
\examples{
node2v_walk(adj = c(1, 2, 3, 4, 5, 6, 7, 8), start = c(1, 2, 3, 4, 5, 6, 7, 8), length = 5L)
}
