% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_km_censoring}
\alias{.morie_km_censoring}
\title{.morie_km_censoring}
\usage{
.morie_km_censoring(x, delta)
}
\arguments{
\item{x}{A vector; its length is taken and its elements indexed.}
\item{delta}{A vector; indexed elementwise.}
}
\description{
Part of the twostg_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{function}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.morie_km_censoring(V, V)
}
