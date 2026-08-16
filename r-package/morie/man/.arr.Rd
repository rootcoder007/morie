% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.arr}
\alias{.arr}
\title{.arr}
\usage{
.arr(x)
}
\arguments{
\item{x}{Coerced to numeric by the body, with \code{as.numeric}.}
}
\description{
A step of the effect_sizes implementation. Called by \code{cles}, \code{cliffs_delta}, \code{coefficient_of_variation} and 7 others in the module. See the file header for the source the module follows.
}
\value{
The value of \code{[}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.arr(V)
}
