% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.svyqtl_inv}
\alias{.svyqtl_inv}
\title{.svyqtl_inv}
\usage{
.svyqtl_inv(xs, cum, p)
}
\arguments{
\item{xs}{A vector; its length is taken and its elements indexed.}
\item{cum}{Passed to \code{>=}.}
\item{p}{Passed to \code{>=}.}
}
\description{
Part of the Svyqtl implementation; see the file header for the source it follows.
}
\value{
One of two values, depending on the branch taken.
}
\keyword{internal}
\examples{
.svyqtl_inv(xs = c(1, 2, 3, 4, 5, 6, 7, 8), cum = c(1, 2, 3, 4, 5, 6, 7, 8), p = 0.5)
}
