% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.smcopt_rwk}
\alias{.smcopt_rwk}
\title{Bare random-walk kernel with a caller-chosen scale.  Mirrors}
\usage{
.smcopt_rwk(scale)
}
\arguments{
\item{scale}{See Usage.}
}
\description{
the Python arm\\\'s random_walk_kernel(scale=1.0): propose x\\\' = x + Normal(0, scale^2) and accept with the standard Metropolis-Hastings ratio.  No library calls; uses .ghc_norm so the run is reproducible from the seed.
}
\value{
The value of \code{function}.
}
\keyword{internal}
