% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.vrgft_obj}
\alias{.vrgft_obj}
\title{Internal: variogram weighted-least-squares objective}
\usage{
.vrgft_obj(p, mids, gammas, weights, model)
}
\arguments{
\item{p}{A vector; indexed elementwise.}
\item{mids}{Passed to \code{.vrgft_model}.}
\item{gammas}{Numeric; combined arithmetically in the body.}
\item{weights}{Numeric; combined arithmetically in the body.}
\item{model}{Passed to \code{.vrgft_model}.}
}
\description{
Part of the vrgft implementation; see the file header for the source it follows.
}
\value{
A numeric value.
}
\keyword{internal}
