% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{miprgr_simplex}
\alias{miprgr_simplex}
\title{Two-phase simplex with Bland\\'s rule, maximising c\\'x}
\usage{
miprgr_simplex(A, b, c, tol = 1e-09, max_iter = 20000)
}
\arguments{
\item{A}{See Usage.}
\item{b}{See Usage.}
\item{c}{See Usage.}
\item{tol}{Defaults to \code{1e-09}.}
\item{max_iter}{Defaults to \code{20000}.}
}
\description{
Part of the miprgr_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{feasible}, \code{x}, \code{value}.
}
\references{
  Land, A. H. & Doig, A. G. (1960) Econometrica 28(3), 497-520.
  Dakin, R. J. (1965) The Computer Journal 8(3), 250-255.
Base R only.
}
\keyword{internal}
\examples{
miprgr_simplex(A = matrix(c(1, 2, 3, 4, 5, 6), nrow = 2), b = 5L, c = matrix(c(1, 2, 3, 4, 5, 6), nrow = 2))
}
