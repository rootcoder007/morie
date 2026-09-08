% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{muzero_add_noise}
\alias{muzero_add_noise}
\title{muzero_add_noise}
\usage{
muzero_add_noise(prior, alpha, frac, seed) 
}
\arguments{
\item{prior}{Numeric; combined arithmetically in the body.}
\item{alpha}{Passed to \code{<=}.}
\item{frac}{Numeric; combined arithmetically in the body.}
\item{seed}{Passed to \code{set.seed}.}
}
\description{
Part of the muzero_native implementation; see the file header for the source it follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
p <- muzero_add_noise(c(0.25, 0.25, 0.25, 0.25), alpha = 0.3,
                      frac = 0.25, seed = 1)
c(length(p), abs(sum(p) - 1) < 1e-9)
}
