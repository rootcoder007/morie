% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.unif_stream}
\alias{.unif_stream}
\title{.unif_stream}
\usage{
.unif_stream(seed, block = 8192L)
}
\arguments{
\item{seed}{Passed to \code{.ghc_rng}.}
\item{block}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{8192L}.}
}
\description{
A step of the bayrjmcmc_native implementation. Called by \code{changepoint_rjmcmc}, \code{reversible_jump_mcmc}. See the file header for the source the module follows.
}
\value{
The value of \code{function}.
}
\keyword{internal}
\examples{
.unif_stream(seed = 5L)
}
