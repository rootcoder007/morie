% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{miprgr_round_relaxation}
\alias{miprgr_round_relaxation}
\title{miprgr_round_relaxation}
\usage{
miprgr_round_relaxation(x, A, b, integer_vars)
}
\arguments{
\item{x}{See Usage.}
\item{A}{A matrix; indexed by row and column.}
\item{b}{A vector; indexed elementwise.}
\item{integer_vars}{See Usage.}
}
\description{
Part of the miprgr_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{x}, \code{feasible}, \code{violations}, \code{note}.
}
\references{
  Land, A. H. & Doig, A. G. (1960) Econometrica 28(3), 497-520.
  Dakin, R. J. (1965) The Computer Journal 8(3), 250-255.
Base R only.
}
\keyword{internal}
\examples{
miprgr_round_relaxation(x = c(1, 2, 3, 4, 5, 6, 7, 8), A = matrix(c(1, 2, 3, 4, 5, 6), nrow = 2), b = c(1, 2, 3, 4, 5, 6, 7, 8), integer_vars = c(1, 2, 3, 4, 5, 6, 7, 8))
}
