% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.sv_isotonic_pava}
\alias{.sv_isotonic_pava}
\title{.sv_isotonic_pava}
\usage{
.sv_isotonic_pava(y, w = NULL)
}
\arguments{
\item{y}{A vector; its length is taken.}
\item{w}{Optional; may be \code{NULL}. Coerced to list by the body, with \code{as.list}.}
}
\description{
Part of the spatial_voting implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.sv_isotonic_pava(V)
}
