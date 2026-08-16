% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_node2v}
\alias{morie_node2v}
\title{morie_node2v}
\usage{
morie_node2v(adj, num_walks = 10, length = 10, p = 1, q = 1, seed = 0, weights = NULL)
}
\arguments{
\item{adj}{A vector; its length is taken.}
\item{num_walks}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{10}.}
\item{length}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{10}.}
\item{p}{Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{1}.}
\item{q}{Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{1}.}
\item{seed}{Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{0}.}
\item{weights}{Passed to \code{node2v_walk}.}
}
\description{
Part of the node2v_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{walks}, \code{p}, \code{q}, \code{n_walks}, \code{length}, \code{method}, \code{note}.
}
\keyword{internal}
