% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{rate}
\alias{rate}
\title{RATE: int alpha(u) TOC(u) du}
\usage{
rate(scores, priority, weight = "autoc")
}
\description{
RATE: int alpha(u) TOC(u) du
}
\value{
A list with \code{estimate}, \code{weight}, \code{curve}, \code{n}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
rate(V, V)
}
