% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.unwrap}
\alias{.unwrap}
\title{Numpy.unwrap port: shift phase jumps > pi by 2*pi}
\usage{
.unwrap(p, tol = pi)
}
\arguments{
\item{p}{A vector; indexed elementwise.}
\item{tol}{Numeric; combined arithmetically in the body. Defaults to \code{pi}.}
}
\description{
A step of the dsp_waveform implementation. Called by \code{morie_dsp_complex_demodulation}. See the file header for the source the module follows.
}
\value{
A vector, from \code{c}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.unwrap(V)
}
