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
\item{M}{See Usage.}
\item{observed}{See Usage.}
\item{tau}{Defaults to \code{NULL}.}
\item{step}{Defaults to \code{1.9}.}
\item{iters}{Defaults to \code{200L}.}
\item{tol}{Defaults to \code{1e-06}.}
}
\description{
Part of the meglt_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{X}, \code{residual_history}, \code{final_residual}, \code{tau}, \code{n_observed}, \code{nuclear_norm}, \code{relative_error}, \code{method}.
}
\keyword{internal}
