% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{error_cache}
\alias{error_cache}
\title{error_cache}
\usage{
error_cache(alpha, y, K, b)
}
\arguments{
\item{alpha}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{y}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{K}{Passed to \code{.smoopt_K}.}
\item{b}{Coerced to numeric by the body, with \code{as.numeric}.}
}
\description{
Part of the smoopt_native implementation; see the file header for the source it follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
error_cache(alpha = 0.5, y = c(1, 2, 3, 4, 5, 6, 7, 8), K = 5L, b = 5L)
}
