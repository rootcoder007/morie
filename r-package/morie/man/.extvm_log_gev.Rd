% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.extvm_log_gev}
\alias{.extvm_log_gev}
\title{Internal: GEV per-observation log-density (Coles xi convention)}
\usage{
.extvm_log_gev(par, x)
}
\arguments{
\item{par}{A vector; indexed elementwise.}
\item{x}{A vector; its length is taken.}
}
\description{
Lifted from the extvm() optimiser closure so the xi ~ 0 (Gumbel) and out-of-support branches are directly unit-testable; the BFGS search in extvm() is not guaranteed to probe xi within 1e-8 of zero.
}
\value{
The value of \code{ll}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.extvm_log_gev(V, V)
}
