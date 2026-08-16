% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{systematic_resample}
\alias{systematic_resample}
\title{Systematic resampling}
\usage{
systematic_resample(weights, u = NULL)
}
\arguments{
\item{weights}{Numeric vector of positive weights.}
\item{u}{Optional fixed offset in \code{[0, 1)}; if \code{NULL} one   uniform is drawn from the shared generator.}
}
\description{
One uniform, J evenly spaced points through the cumulative weights; the count particle j receives differs from \code{J * w_j} by less than one, deterministically.
}
\value{
Integer vector of indices.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
systematic_resample(V)
}
