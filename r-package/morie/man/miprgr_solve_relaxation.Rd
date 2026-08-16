% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{miprgr_solve_relaxation}
\alias{miprgr_solve_relaxation}
\title{miprgr_solve_relaxation}
\usage{
miprgr_solve_relaxation(A, b, c, bounds = list(), n = NULL, maximise = TRUE, solver = "simplex")
}
\arguments{
\item{A}{See Usage.}
\item{b}{See Usage.}
\item{c}{See Usage.}
\item{bounds}{Defaults to \code{list()}.}
\item{n}{Defaults to \code{NULL}.}
\item{maximise}{Defaults to \code{TRUE}.}
\item{solver}{Defaults to \code{"simplex"}.}
}
\description{
Part of the miprgr_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{feasible}, \code{x}, \code{value}, \code{note}.
}
\references{
  Land, A. H. & Doig, A. G. (1960) Econometrica 28(3), 497-520.
  Dakin, R. J. (1965) The Computer Journal 8(3), 250-255.
Base R only.
}
\keyword{internal}
\examples{
miprgr_solve_relaxation(A = matrix(c(1, 2, 3, 4, 5, 6), nrow = 2), b = 5L, c = matrix(c(1, 2, 3, 4, 5, 6), nrow = 2))
}
