% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{kernel_hac}
\alias{kernel_hac}
\title{Equation 2.3, kernel HAC on already-prewhitened vectors}
\usage{
kernel_hac(v, bandwidth, kernel = "qs", n_params = 0L, n = NULL)
}
\description{
Equation 2.3, kernel HAC on already-prewhitened vectors
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
kernel_hac(v = c(1, 2, 3, 4, 5, 6, 7, 8), bandwidth = 0.5)
}
