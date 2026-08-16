% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{synthesize_marginals}
\alias{synthesize_marginals}
\title{synthesize_marginals}
\usage{
synthesize_marginals(X, n, seed = 0)
}
\arguments{
\item{X}{A vector; its length is taken and its elements indexed.}
\item{n}{See Usage.}
\item{seed}{Passed to \code{.memb_rng}. Defaults to \code{0}.}
}
\description{
Part of the memb_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{split}.
}
\keyword{internal}
\examples{
synthesize_marginals(X = 5L, n = 5L)
}
