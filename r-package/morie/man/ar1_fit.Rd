% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{ar1_fit}
\alias{ar1_fit}
\title{AR(1) by least squares (no intercept)}
\usage{
ar1_fit(x) 
}
\description{
AR(1) by least squares (no intercept)
}
\arguments{
\item{x}{Argument \code{x}; see Usage.}
}
\value{
A vector.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
ar1_fit(V)
}
