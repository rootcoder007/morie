% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.otis_multiway_cluster_se}
\alias{.otis_multiway_cluster_se}
\title{Cameron-Gelbach-Miller multi-way cluster-robust SE (up to 2-way)}
\usage{
.otis_multiway_cluster_se(scores, clusters)
}
\arguments{
\item{scores}{Passed to \code{.otis_cluster_se}.}
\item{clusters}{A vector; its length is taken and its elements indexed.}
}
\description{
Part of the otis_causal implementation; see the file header for the source it follows.
}
\value{
The value of \code{.otis_cluster_se}.
}
\keyword{internal}
\examples{
.otis_multiway_cluster_se(scores = 5L, clusters = c(1, 2, 3, 4, 5, 6, 7, 8))
}
