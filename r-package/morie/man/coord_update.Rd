% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{coord_update}
\alias{coord_update}
\title{coord_update}
\usage{
coord_update(X, M, phi_x, C = NULL)
}
\arguments{
\item{X}{A matrix; indexed by row and column.}
\item{M}{A vector; indexed elementwise.}
\item{phi_x}{See Usage.}
\item{C}{Optional; may be \code{NULL}. Coerced to numeric by the body, with \code{as.numeric}.}
}
\description{
A step of the egnnL_native implementation. Called by \code{egcl}. See the file header for the source the module follows.
}
\value{
The value of \code{lapply}.
}
\keyword{internal}
