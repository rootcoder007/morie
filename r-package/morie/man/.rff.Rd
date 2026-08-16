% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.rff}
\alias{.rff}
\title{.rff}
\usage{
.rff(M, W1, b1, W2, b2)
}
\arguments{
\item{M}{A matrix; passed to \code{\%*\%}.}
\item{W1}{A matrix; passed to \code{\%*\%}.}
\item{b1}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{W2}{A matrix; passed to \code{\%*\%}.}
\item{b2}{Coerced to numeric by the body, with \code{as.numeric}.}
}
\description{
Part of the setT_mixedcase_native implementation; see the file header for the source it follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
.rff(M = 5L, W1 = c(1, 2, 3, 4, 5, 6, 7, 8), b1 = c(1, 2, 3, 4, 5, 6, 7, 8), W2 = c(1, 2, 3, 4, 5, 6, 7, 8), b2 = c(1, 2, 3, 4, 5, 6, 7, 8))
}
