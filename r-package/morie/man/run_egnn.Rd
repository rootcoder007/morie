% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{run_egnn}
\alias{run_egnn}
\title{run_egnn}
\usage{
run_egnn(H, X, layers, phi_e, phi_x, phi_h, A = NULL, C = NULL)
}
\arguments{
\item{H}{A matrix; passed to \code{as.matrix}.}
\item{X}{A matrix; passed to \code{as.matrix}.}
\item{layers}{Coerced to integer by the body, with \code{as.integer}.}
\item{phi_e}{Passed to \code{egcl}.}
\item{phi_x}{Passed to \code{egcl}.}
\item{phi_h}{Passed to \code{egcl}.}
\item{A}{Passed to \code{egcl}.}
\item{C}{Passed to \code{egcl}.}
}
\description{
A step of the egnnL_native implementation. Called by \code{egnn_layer}, \code{egnnlayer}, \code{equivariantgnn} and 2 others in the module. See the file header for the source the module follows.
}
\value{
A list with \code{estimate}, \code{H}, \code{X}, \code{layers}, \code{method}, \code{note}.
}
\keyword{internal}
