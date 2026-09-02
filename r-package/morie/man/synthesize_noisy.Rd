% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{synthesize_noisy}
\alias{synthesize_noisy}
\title{synthesize_noisy}
\usage{
synthesize_noisy(X, fraction = 0.1, feature_values = NULL, seed = 0) 
}
\arguments{
\item{X}{A vector; indexed elementwise.}
\item{fraction}{The body requires: memb: fraction must lie in [0, 1]. Defaults to \code{0.1}.}
\item{feature_values}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
\item{seed}{Passed to \code{.memb_rng}. Defaults to \code{0}.}
}
\description{
Part of the memb_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
synthesize_noisy(V)
}
