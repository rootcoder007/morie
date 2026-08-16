% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.hawkes_restarts}
\alias{.hawkes_restarts}
\title{Deterministic multi-start set in the unconstrained space. No RNG, so}
\usage{
.hawkes_restarts(phi0)
}
\arguments{
\item{phi0}{A vector; its length is taken.}
}
\description{
the fit is reproducible regardless of the caller\\'s random seed. The perturbations span lower / higher eta and shifted baseline / shape; the lower-eta start in particular lets the optimiser reach the Poisson submodel when the data carries no self-excitation.
}
\value{
The value of \code{lapply}.
}
\keyword{internal}
