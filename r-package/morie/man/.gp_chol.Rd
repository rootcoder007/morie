% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.gp_chol}
\alias{.gp_chol}
\title{.gp_chol}
\usage{
.gp_chol(a, jitter = 1e-12)
}
\arguments{
\item{a}{A matrix; indexed by row and column.}
\item{jitter}{Defaults to \code{1e-12}.}
}
\description{
A step of the abcgp_native implementation. Called by \code{.gp_draw_mean}, \code{.gp_mvn_logpdf}, \code{.gp_profile_nll} and 1 others in the module. See the file header for the source the module follows.
}
\value{
The value of \code{L}, as built in the body.
}
\keyword{internal}
\examples{
M <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
.gp_chol(M)
}
