% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{sobol_sequence}
\alias{sobol_sequence}
\title{Sobol low-discrepancy sequence in [0, 1)^dim}
\usage{
sobol_sequence(n, dim, skip = 0L)
}
\arguments{
\item{n}{Number of points, integer >= 1.}
\item{dim}{Dimension; must be in 1..8 (the tabulated direction   numbers).}
\item{skip}{Drop that many leading points.}
}
\description{
The first \code{n} points of the Sobol sequence in the requested dimension. The design is shared with the Python arm down to the last bit and is the design Wilkinson (2014) Sec. 2.2 uses.
}
\value{
Numeric matrix with \code{n} rows and \code{dim} columns.
}
\keyword{internal}
\examples{
s <- sobol_sequence(8, 2)
c(nrow(s), ncol(s), all(s >= 0 & s <= 1))
}
