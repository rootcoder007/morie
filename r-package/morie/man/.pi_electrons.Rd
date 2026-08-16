% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.pi_electrons}
\alias{.pi_electrons}
\title{Pi-electron count per atom, rounded to an integer}
\usage{
.pi_electrons(n_atoms, bonds)
}
\arguments{
\item{n_atoms}{A count; the body uses it as \code{rep(...)}.}
\item{bonds}{See Usage.}
}
\description{
Part of the toptor_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{as.integer}.
}
\keyword{internal}
\examples{
.pi_electrons(n_atoms = 5L, bonds = data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9)))
}
