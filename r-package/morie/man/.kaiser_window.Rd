% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.kaiser_window}
\alias{.kaiser_window}
\title{Kaiser window with shape `beta` using the modified Bessel function}
\usage{
.kaiser_window(N, beta = 14)
}
\arguments{
\item{N}{Numeric; combined arithmetically in the body.}
\item{beta}{Numeric; combined arithmetically in the body. Defaults to \code{14}.}
}
\description{
I0. Matches numpy.kaiser(N, beta).
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
.kaiser_window(N = 5L)
}
