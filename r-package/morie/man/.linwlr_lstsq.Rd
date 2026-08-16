% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.linwlr_lstsq}
\alias{.linwlr_lstsq}
\title{.linwlr_lstsq}
\usage{
.linwlr_lstsq(Z, y, ridge)
}
\arguments{
\item{Z}{A matrix; passed to \code{ncol}.}
\item{y}{A matrix; passed to \code{crossprod}.}
\item{ridge}{Numeric; combined arithmetically in the body.}
}
\description{
Part of the linwlr_native implementation; see the file header for the source it follows.
}
\value{
A matrix, from \code{solve}.
}
\keyword{internal}
\examples{
.linwlr_lstsq(Z = matrix(c(1, 2, 3, 4, 5, 6), nrow = 2), y = matrix(c(1, 2, 3, 4, 5, 6), nrow = 2), ridge = c(1, 2, 3, 4, 5, 6, 7, 8))
}
