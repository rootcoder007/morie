% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{node2v_transition_probabilities}
\alias{node2v_transition_probabilities}
\title{node2v_transition_probabilities}
\usage{
node2v_transition_probabilities(adj, t, v, p, q, weights = NULL)
}
\arguments{
\item{adj}{A vector; indexed elementwise.}
\item{t}{See Usage.}
\item{v}{Coerced to character by the body, with \code{as.character}.}
\item{p}{See Usage.}
\item{q}{See Usage.}
\item{weights}{Optional; may be \code{NULL}. A vector; indexed elementwise.}
}
\description{
Part of the node2v_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{nodes}, \code{probabilities}, \code{unnormalized}, \code{Z}.
}
\keyword{internal}
