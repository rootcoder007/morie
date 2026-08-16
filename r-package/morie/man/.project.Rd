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
\item{values}{See Usage.}
\item{basis}{See Usage.}
\item{n}{See Usage.}
\item{ridge}{Passed to \code{.tmldyn_lstsq}.}
}
\description{
Part of the tmldyn_native implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{as.numeric}.
}
\keyword{internal}
