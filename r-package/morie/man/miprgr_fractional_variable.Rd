% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{miprgr_fractional_variable}
\alias{miprgr_fractional_variable}
\title{miprgr_fractional_variable}
\usage{
miprgr_fractional_variable(x, integer_vars, tol = 1e-07)
}
\arguments{
\item{x}{A vector; indexed elementwise.}
\item{integer_vars}{See Usage.}
\item{tol}{Defaults to \code{1e-07}.}
}
\description{
Part of the miprgr_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{index}, \code{fractionality}, \code{integral}.
}
\references{
  Land, A. H. & Doig, A. G. (1960) Econometrica 28(3), 497-520.
  Dakin, R. J. (1965) The Computer Journal 8(3), 250-255.
Base R only.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
miprgr_fractional_variable(V, V)
}
