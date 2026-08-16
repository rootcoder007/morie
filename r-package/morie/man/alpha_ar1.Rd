% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{alpha_ar1}
\alias{alpha_ar1}
\title{Andrews (1991) eq. 6.4, alpha(q) from p AR(1) fits}
\usage{
alpha_ar1(v, q = 2L, weights = NULL)
}
\description{
Andrews (1991) eq. 6.4, alpha(q) from p AR(1) fits
}
\value{
A list with \code{alpha}, \code{fits}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
alpha_ar1(V)
}
