% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.ols_slope}
\alias{.ols_slope}
\title{.ols_slope}
\usage{
.ols_slope(xs, ys)
}
\arguments{
\item{xs}{A vector; its length is taken.}
\item{ys}{Numeric; passed to \code{mean}.}
}
\description{
Part of the lyapun_native implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{c}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.ols_slope(V, V)
}
