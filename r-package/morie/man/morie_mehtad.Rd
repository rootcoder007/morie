% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_mehtad}
\alias{morie_mehtad}
\title{morie_mehtad}
\usage{
morie_mehtad(A, b, c, tol = 1e-09, max_iter = 100L, nu = 3, eta = 0.9995, corrector = TRUE)
}
\arguments{
\item{A}{Passed to \code{solve_lp}.}
\item{b}{Passed to \code{solve_lp}.}
\item{c}{Passed to \code{solve_lp}.}
\item{tol}{Passed to \code{solve_lp}. Defaults to \code{1e-09}.}
\item{max_iter}{Passed to \code{solve_lp}. Defaults to \code{100L}.}
\item{nu}{Passed to \code{solve_lp}. Defaults to \code{3}.}
\item{eta}{Passed to \code{solve_lp}. Defaults to \code{0.9995}.}
\item{corrector}{Passed to \code{solve_lp}. Defaults to \code{TRUE}.}
}
\description{
Part of the mehtad_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{solve_lp}.
}
\keyword{internal}
