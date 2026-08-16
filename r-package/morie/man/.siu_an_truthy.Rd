% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.siu_an_truthy}
\alias{.siu_an_truthy}
\title{.siu_an_truthy}
\usage{
.siu_an_truthy(v)
}
\arguments{
\item{v}{A logical; the body checks with \code{is.logical}.}
}
\description{
Truthy / falsy counters tolerant of CSV-roundtripped booleans. ---------------------------------------------------------------------------
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.siu_an_truthy(V)
}
