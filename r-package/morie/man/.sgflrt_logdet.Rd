% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.sgflrt_logdet}
\alias{.sgflrt_logdet}
\title{.sgflrt_logdet}
\usage{
.sgflrt_logdet(L)
}
\arguments{
\item{L}{A matrix; passed to \code{diag}.}
}
\description{
Part of the sgflrt_native implementation; see the file header for the source it follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.sgflrt_logdet(V)
}
