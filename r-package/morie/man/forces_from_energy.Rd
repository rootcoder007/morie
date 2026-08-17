% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{forces_from_energy}
\alias{forces_from_energy}
\title{forces_from_energy}
\usage{
forces_from_energy(energy_fn, R, h = 1e-05)
}
\arguments{
\item{energy_fn}{Accepted by the signature and not used anywhere in the body.}
\item{R}{Iterated over elementwise, with \code{lapply}.}
\item{h}{Numeric; combined arithmetically in the body. Defaults to \code{1e-05}.}
}
\description{
Part of the schN_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{forces}, \code{net_force}, \code{method}, \code{note}.
}
\keyword{internal}
\examples{
set.seed(1)
R <- lapply(1:4, function(i) rnorm(3))
energy_fn <- function(pos) sum(vapply(pos, function(p) sum(p^2), 0))
Fm <- forces_from_energy(energy_fn, R)
length(Fm$forces) == 4L
}
