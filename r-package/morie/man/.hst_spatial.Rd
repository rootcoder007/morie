% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.hst_spatial}
\alias{.hst_spatial}
\title{Gaussian spatial density evaluated at squared distances}
\usage{
.hst_spatial(d2, sigma)
}
\arguments{
\item{d2}{Numeric; combined arithmetically in the body.}
\item{sigma}{Numeric; combined arithmetically in the body.}
}
\description{
A step of the hawkes_spatial implementation. Called by \code{morie_hawkes_st_intensity}, \code{morie_hawkes_st_loglik}. See the file header for the source the module follows.
}
\value{
A numeric value.
}
\keyword{internal}
