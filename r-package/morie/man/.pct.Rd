% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.pct}
\alias{.pct}
\title{Helper: percentile-of-vector (matches numpy.percentile linear interp)}
\usage{
.pct(x, p)
}
\arguments{
\item{x}{See Usage.}
\item{p}{Numeric; combined arithmetically in the body.}
}
\description{
A step of the bootstrap_methods implementation. Called by \code{.bca_interval}, \code{block_bootstrap}, \code{bootstrap} and 3 others in the module. See the file header for the source the module follows.
}
\value{
The value of \code{unname}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.pct(V, V)
}
