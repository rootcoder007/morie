% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.gp_mle_lengthscale}
\alias{.gp_mle_lengthscale}
\title{.gp_mle_lengthscale}
\usage{
.gp_mle_lengthscale(X, y, nug, kernel)
}
\arguments{
\item{X}{A matrix; indexed by row and column.}
\item{y}{Passed to \code{.gp_profile_nll}.}
\item{nug}{Passed to \code{.gp_profile_nll}.}
\item{kernel}{Passed to \code{.gp_profile_nll}.}
}
\description{
A step of the abcgp_native implementation. Called by \code{gp_fit}. See the file header for the source the module follows.
}
\value{
The value of \code{ls}, as built in the body.
}
\keyword{internal}
