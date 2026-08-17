% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{invariance_error}
\alias{invariance_error}
\title{invariance_error}
\usage{
invariance_error(energy_fn, R, Q, g = NULL)
}
\arguments{
\item{energy_fn}{Passed to \code{forces_from_energy}.}
\item{R}{Iterated over elementwise, with \code{lapply}.}
\item{Q}{A matrix; passed to \code{as.matrix}.}
\item{g}{Optional; may be \code{NULL}. Coerced to numeric by the body, with \code{as.numeric}.}
}
\description{
Part of the schN_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{energy_error}, \code{force_error}, \code{energy_invariant}, \code{forces_equivariant}, \code{note}.
}
\keyword{internal}
\examples{
set.seed(1)
R <- lapply(1:4, function(i) rnorm(3))
energy_fn <- function(pos) sum(vapply(pos, function(p) sum(p^2), 0))
th <- pi / 4
Q <- matrix(c(cos(th), -sin(th), 0, sin(th), cos(th), 0, 0, 0, 1), 3, 3)
ie <- invariance_error(energy_fn, R, Q)
is.list(ie) || is.numeric(ie)
}
