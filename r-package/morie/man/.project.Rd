% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.project}
\alias{.project}
\title{Least-squares projection of values onto the basis V at the eval rows}
\usage{
.project(values, basis, n, ridge)
}
\arguments{
\item{values}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{basis}{Optional; may be \code{NULL}. A matrix; the body checks with \code{is.matrix}.}
\item{n}{Accepted by the signature and not used anywhere in the body.}
\item{ridge}{Passed to \code{.tmldyn_lstsq}.}
}
\description{
Part of the tmldyn_native implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{as.numeric}.
}
\keyword{internal}
