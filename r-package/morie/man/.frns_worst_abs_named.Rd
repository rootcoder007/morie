% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.frns_worst_abs_named}
\alias{.frns_worst_abs_named}
\title{Name of the element with the largest absolute value}
\usage{
.frns_worst_abs_named(x)
}
\arguments{
\item{x}{A vector; its length is taken and its elements indexed.}
}
\description{
A step of the frns_predpol implementation. Called by \code{morie_predpol_calibration_audit}. See the file header for the source the module follows.
}
\value{
The value of \code{[}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.frns_worst_abs_named(V)
}
