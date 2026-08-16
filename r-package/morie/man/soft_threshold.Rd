% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{soft_threshold}
\alias{soft_threshold}
\title{Soft-thresholding}
\usage{
soft_threshold(v, tau)
}
\arguments{
\item{v}{Numeric vector.}
\item{tau}{Non-negative threshold.}
}
\description{
Prox of \code{tau * |.|_1}, applied elementwise.
}
\value{
Numeric vector of the same length.
}
\keyword{internal}
