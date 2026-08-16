% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.fisher_z_ci}
\alias{.fisher_z_ci}
\title{.fisher_z_ci}
\usage{
.fisher_z_ci(r, n, confidence)
}
\arguments{
\item{r}{Passed to \code{atanh}.}
\item{n}{Numeric; combined arithmetically in the body.}
\item{confidence}{Numeric; combined arithmetically in the body.}
}
\description{
Part of the statistics implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{c}.
}
\keyword{internal}
\examples{
.fisher_z_ci(r = c(1, 2, 3, 4, 5, 6, 7, 8), n = 5L, confidence = c(1, 2, 3, 4, 5, 6, 7, 8))
}
