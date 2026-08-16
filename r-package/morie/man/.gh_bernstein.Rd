% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.gh_bernstein}
\alias{.gh_bernstein}
\title{.gh_bernstein}
\usage{
.gh_bernstein(u, K)
}
\arguments{
\item{u}{A vector; its length is taken.}
\item{K}{A count; the body uses it as \code{seq_len(...)}.}
}
\description{
A step of the helpers_ghosal_bnp implementation. Called by \code{morie_ghosal_sieve_prior}. See the file header for the source the module follows.
}
\value{
The value of \code{B}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.gh_bernstein(V, V)
}
