% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.gp_profile_nll}
\alias{.gp_profile_nll}
\title{.gp_profile_nll}
\usage{
.gp_profile_nll(X, y, ls, nug, kernel)
}
\arguments{
\item{X}{A matrix; indexed by row and column.}
\item{y}{Numeric; combined arithmetically in the body.}
\item{ls}{Passed to \code{.gp_corr}.}
\item{nug}{A vector; indexed elementwise.}
\item{kernel}{Passed to \code{.gp_corr}.}
}
\description{
A step of the abcgp_native implementation. Called by \code{.gp_mle_lengthscale}. See the file header for the source the module follows.
}
\value{
A numeric value.
}
\keyword{internal}
