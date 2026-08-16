% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.gp_chol_solve}
\alias{.gp_chol_solve}
\title{.gp_chol_solve}
\usage{
.gp_chol_solve(L, b)
}
\arguments{
\item{L}{A matrix; indexed by row and column.}
\item{b}{A vector; indexed elementwise.}
}
\description{
A step of the abcgp_native implementation. Called by \code{.gp_mvn_logpdf}, \code{.gp_profile_nll}, \code{gp_fit} and 1 others in the module. See the file header for the source the module follows.
}
\value{
The value of \code{x}, as built in the body.
}
\keyword{internal}
