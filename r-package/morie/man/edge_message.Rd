% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{edge_message}
\alias{edge_message}
\title{edge_message}
\usage{
edge_message(h_i, h_j, x_i, x_j, phi_e, a_ij = NULL)
}
\arguments{
\item{h_i}{See Usage.}
\item{h_j}{See Usage.}
\item{x_i}{See Usage.}
\item{x_j}{See Usage.}
\item{phi_e}{See Usage.}
\item{a_ij}{Defaults to \code{NULL}.}
}
\description{
A step of the egnnL_native implementation. Called by \code{egcl}. See the file header for the source the module follows.
}
\value{
The value of \code{phi_e}.
}
\keyword{internal}
