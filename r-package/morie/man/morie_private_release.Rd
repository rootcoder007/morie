% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_private_release}
\alias{morie_private_release}
\title{Laplace mechanism release}
\usage{
morie_private_release(value, sensitivity, epsilon, seed = 0)
}
\arguments{
\item{value}{The (non-private) output to release.}
\item{sensitivity}{Positive l1 sensitivity of \code{f}.}
\item{epsilon}{Privacy parameter.}
\item{seed}{Seed for the shared generator.}
}
\description{
Returns \code{f(D) + Lap(sensitivity/epsilon)} from the shared generator so the R and Python arms produce the same noise.
}
\value{
A list with the released value, the noise, the scale, the   noise variance, and the epsilon.
}
\keyword{internal}
\examples{
r <- morie_private_release(value = 2.5, sensitivity = 0.1,
                           epsilon = 1, seed = 1)
c(is.numeric(r$noise_variance), r$noise_variance > 0)
}
