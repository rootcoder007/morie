% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.scalar}
\alias{.scalar}
\title{.scalar}
\usage{
.scalar(state)
}
\arguments{
\item{state}{A vector; indexed elementwise.}
}
\description{
Part of the prtcl_native implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{as.numeric}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.scalar(V)
}
