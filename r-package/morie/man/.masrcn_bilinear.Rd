% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.masrcn_bilinear}
\alias{.masrcn_bilinear}
\title{.masrcn_bilinear}
\usage{
.masrcn_bilinear(F, y, x)
}
\arguments{
\item{F}{A matrix; indexed by row and column.}
\item{y}{Numeric; combined arithmetically in the body.}
\item{x}{Numeric; combined arithmetically in the body.}
}
\description{
Part of the masrcn_native implementation; see the file header for the source it follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
.masrcn_bilinear(F = matrix(c(1, 2, 3, 4, 5, 6), nrow = 2), y = c(1, 2, 3, 4, 5, 6, 7, 8), x = c(1, 2, 3, 4, 5, 6, 7, 8))
}
