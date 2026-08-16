% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.gp_expected_error}
\alias{.gp_expected_error}
\title{.gp_expected_error}
\usage{
.gp_expected_error(alphas, tau, n_grid = 101L)
}
\arguments{
\item{alphas}{A vector; its length is taken.}
\item{tau}{Passed to \code{<=}.}
\item{n_grid}{A count; the body uses it as \code{seq_len(...)}. Defaults to \code{101L}.}
}
\description{
A step of the abcgp_native implementation. Called by \code{.gp_mw_sampler}. See the file header for the source the module follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
.gp_expected_error(alphas = c(1, 2, 3, 4, 5, 6, 7, 8), tau = 0.5)
}
