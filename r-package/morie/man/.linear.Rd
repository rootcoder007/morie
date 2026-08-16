% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.linear}
\alias{.linear}
\title{.linear}
\usage{
.linear(x, Wm, b)
}
\arguments{
\item{x}{A vector; its length is taken and its elements indexed.}
\item{Wm}{A matrix; indexed by row and column.}
\item{b}{A vector; indexed elementwise.}
}
\description{
Part of the mambss_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
\examples{
.linear(x = 5L, Wm = matrix(c(1, 2, 3, 4, 5, 6), nrow = 2), b = 5L)
}
