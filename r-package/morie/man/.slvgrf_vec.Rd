% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.slvgrf_vec}
\alias{.slvgrf_vec}
\title{.slvgrf_vec}
\usage{
.slvgrf_vec(x)
}
\arguments{
\item{x}{A matrix; passed to \code{as.matrix}.}
}
\description{
Part of the slvgrf_native implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{as.numeric}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.slvgrf_vec(V)
}
