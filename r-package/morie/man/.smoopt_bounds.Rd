% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.smoopt_bounds}
\alias{.smoopt_bounds}
\title{Analytic step bounds L <= a2_new <= H.  Mirrors}
\usage{
.smoopt_bounds(i1, i2, alpha, y, C)
}
\arguments{
\item{i1}{See Usage.}
\item{i2}{See Usage.}
\item{alpha}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{y}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{C}{Numeric; passed to \code{min}.}
}
\description{
svmopt._bounds exactly.
}
\value{
A vector, from \code{c}.
}
\keyword{internal}
\examples{
.smoopt_bounds(i1 = 5L, i2 = 5L, alpha = 0.5, y = c(1, 2, 3, 4, 5, 6, 7, 8), C = c(1, 2, 3, 4, 5, 6, 7, 8))
}
