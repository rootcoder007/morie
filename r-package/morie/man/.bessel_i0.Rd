% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.bessel_i0}
\alias{.bessel_i0}
\title{Polynomial expansion of I0 valid for beta up to ~16}
\usage{
.bessel_i0(x)
}
\arguments{
\item{x}{Numeric; passed to \code{abs}.}
}
\description{
A step of the dsp_spectral implementation. Called by \code{.kaiser_window}. See the file header for the source the module follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.bessel_i0(V)
}
