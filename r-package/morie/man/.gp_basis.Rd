% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.gp_basis}
\alias{.gp_basis}
\title{.gp_basis}
\usage{
.gp_basis(theta)
}
\arguments{
\item{theta}{Numeric; combined arithmetically in the body.}
}
\description{
A step of the abcgp_native implementation. Called by \code{gp_predict}. See the file header for the source the module follows.
}
\value{
A vector, from \code{c}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.gp_basis(V)
}
