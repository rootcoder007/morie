% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_laplace_noise}
\alias{morie_laplace_noise}
\title{One draw from Lap(0, b) by inverse transform}
\usage{
morie_laplace_noise(scale, rng) 
}
\arguments{
\item{scale}{Positive scale parameter.}
\item{rng}{Environment produced by \code{.ghc_rng}.}
}
\description{
One draw from Lap(0, b) by inverse transform
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
morie_laplace_noise(scale = TRUE, rng = list(a = 1, b = 2))
}
