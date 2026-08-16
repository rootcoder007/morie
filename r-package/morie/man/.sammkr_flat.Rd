% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.sammkr_flat}
\alias{.sammkr_flat}
\title{.sammkr_flat}
\usage{
.sammkr_flat(m)
}
\arguments{
\item{m}{A matrix; the body checks with \code{is.matrix}.}
}
\description{
Part of the sammkr_native implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{as.numeric}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.sammkr_flat(V)
}
