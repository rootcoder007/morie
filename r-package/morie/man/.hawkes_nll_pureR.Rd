% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.hawkes_nll_pureR}
\alias{.hawkes_nll_pureR}
\title{.hawkes_nll_pureR}
\usage{
.hawkes_nll_pureR(theta, times, end_time, kernel)
}
\arguments{
\item{theta}{A vector; indexed elementwise.}
\item{times}{A vector; its length is taken and its elements indexed.}
\item{end_time}{Numeric; combined arithmetically in the body.}
\item{kernel}{Passed to \code{.hawkes_kernel_funs}.}
}
\description{
A step of the hawkes_fit implementation. Called by \code{morie_hawkes_fit}. See the file header for the source the module follows.
}
\value{
A numeric value.
}
\keyword{internal}
