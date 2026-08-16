% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.tmldyk_lstsq}
\alias{.tmldyk_lstsq}
\title{Solve a least-squares problem with a ridge}
\usage{
.tmldyk_lstsq(Z, yv, ridge = 1e-08)
}
\arguments{
\item{Z}{See Usage.}
\item{yv}{See Usage.}
\item{ridge}{Defaults to \code{1e-08}.}
}
\description{
Part of the tmldyk_native implementation; see the file header for the source it follows.
}
\value{
A matrix, from \code{solve}.
}
\keyword{internal}
