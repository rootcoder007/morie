% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_tcls_dist}
\alias{.morie_tcls_dist}
\title{.morie_tcls_dist}
\usage{
.morie_tcls_dist(labels, domain)
}
\arguments{
\item{labels}{A vector; its length is taken.}
\item{domain}{A vector; its length is taken.}
}
\description{
Part of the tcls_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{p}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.morie_tcls_dist(V, V)
}
