% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.otis_cluster_se}
\alias{.otis_cluster_se}
\title{Liang-Zeger one-way cluster-robust SE for mean of a score vector}
\usage{
.otis_cluster_se(scores, cluster)
}
\arguments{
\item{scores}{A vector; its length is taken.}
\item{cluster}{See Usage.}
}
\description{
Part of the otis_causal implementation; see the file header for the source it follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.otis_cluster_se(V, V)
}
