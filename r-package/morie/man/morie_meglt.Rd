% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_meglt}
\alias{morie_meglt}
\title{morie_meglt}
\usage{
morie_meglt(M, observed, tau = NULL, step = 1.9, iters = 200L, tol = 1e-06) 
}
\arguments{
\item{M}{Passed to \code{svt}.}
\item{observed}{Passed to \code{svt}.}
\item{tau}{Passed to \code{svt}.}
\item{step}{Passed to \code{svt}. Defaults to \code{1.9}.}
\item{iters}{Passed to \code{svt}. Defaults to \code{200L}.}
\item{tol}{Passed to \code{svt}. Defaults to \code{1e-06}.}
}
\description{
Part of the meglt_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{X}, \code{residual_history}, \code{final_residual}, \code{tau}, \code{n_observed}, \code{nuclear_norm}, \code{relative_error}, \code{method}.
}
\keyword{internal}
\examples{
set.seed(2)
U <- matrix(rnorm(10), 5, 2)
V <- matrix(rnorm(8), 4, 2)
M <- U \%*\% t(V)
obs <- matrix(runif(20) < 0.7, 5, 4)
r <- morie_meglt(M, obs)
str(r, max.level = 1)
}
