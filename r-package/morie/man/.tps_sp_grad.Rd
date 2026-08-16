% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.tps_sp_grad}
\alias{.tps_sp_grad}
\title{Central-difference gradient with periodic wrap (gx, gy)}
\usage{
.tps_sp_grad(F_, dx, dy)
}
\arguments{
\item{F_}{Passed to \code{.tps_sp_roll}.}
\item{dx}{Numeric; combined arithmetically in the body.}
\item{dy}{Numeric; combined arithmetically in the body.}
}
\description{
Part of the tps_statphysics implementation; see the file header for the source it follows.
}
\value{
A list with \code{gx}, \code{gy}.
}
\keyword{internal}
\examples{
.tps_sp_grad(F_ = data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9)), dx = c(1, 2, 3, 4, 5, 6, 7, 8), dy = c(1, 2, 3, 4, 5, 6, 7, 8))
}
