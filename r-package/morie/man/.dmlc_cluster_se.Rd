% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.dmlc_cluster_se}
\alias{.dmlc_cluster_se}
\title{Liang-Zeger one-way cluster-robust SE of a mean, from the influence}
\usage{
.dmlc_cluster_se(infl, cluster, n)
}
\arguments{
\item{infl}{Passed to \code{tapply}.}
\item{cluster}{Passed to \code{tapply}.}
\item{n}{Numeric; combined arithmetically in the body.}
}
\description{
function Part of the dml_clustered implementation; see the file header for the source it follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
.dmlc_cluster_se(infl = c(1, 2, 3, 4, 5, 6, 7, 8), cluster = c(1, 2, 3, 4, 5, 6, 7, 8), n = 5L)
}
