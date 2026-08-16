% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.gpfit_log_gp}
\alias{.gpfit_log_gp}
\title{Internal: generalised-Pareto per-exceedance log-density. Lifted from}
\usage{
.gpfit_log_gp(par, y)
}
\arguments{
\item{par}{A vector; indexed elementwise.}
\item{y}{A vector; its length is taken.}
}
\description{
the gpfit() optimiser closure so the xi ~ 0 (exponential) and out-of-support branches are directly unit-testable.
}
\value{
The value of \code{ll}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.gpfit_log_gp(V, V)
}
