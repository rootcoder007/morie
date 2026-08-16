% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.tps_sp_lap}
\alias{.tps_sp_lap}
\title{Periodic 5-point Laplacian}
\usage{
.tps_sp_lap(F_, dx, dy)
}
\arguments{
\item{F_}{Numeric; combined arithmetically in the body.}
\item{dx}{Numeric; passed to \code{max}.}
\item{dy}{Numeric; passed to \code{max}.}
}
\description{
Part of the tps_statphysics implementation; see the file header for the source it follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
.tps_sp_lap(F_ = matrix(c(1, 2, 3, 4, 5, 6), nrow = 2), dx = c(1, 2, 3, 4, 5, 6, 7, 8), dy = c(1, 2, 3, 4, 5, 6, 7, 8))
}
