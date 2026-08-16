% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.attend}
\alias{.attend}
\title{.attend}
\usage{
.attend(X, Y, Wq, Wk, Wv)
}
\arguments{
\item{X}{A matrix; passed to \code{\%*\%}.}
\item{Y}{A matrix; passed to \code{\%*\%}.}
\item{Wq}{A matrix; passed to \code{\%*\%}.}
\item{Wk}{A matrix; passed to \code{\%*\%}.}
\item{Wv}{A matrix; passed to \code{\%*\%}.}
}
\description{
Part of the setT_mixedcase_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{O}, \code{W}.
}
\keyword{internal}
\examples{
.attend(X = c(1, 2, 3, 4, 5, 6, 7, 8), Y = c(1, 2, 3, 4, 5, 6, 7, 8), Wq = c(1, 2, 3, 4, 5, 6, 7, 8), Wk = c(1, 2, 3, 4, 5, 6, 7, 8), Wv = c(1, 2, 3, 4, 5, 6, 7, 8))
}
