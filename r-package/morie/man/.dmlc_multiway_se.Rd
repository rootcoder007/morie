% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.dmlc_multiway_se}
\alias{.dmlc_multiway_se}
\title{Cameron-Gelbach-Miller up to two-way}
\usage{
.dmlc_multiway_se(infl, clusters, n)
}
\arguments{
\item{infl}{Passed to \code{.dmlc_cluster_se}.}
\item{clusters}{A vector; its length is taken and its elements indexed.}
\item{n}{Passed to \code{.dmlc_cluster_se}.}
}
\description{
A step of the dml_clustered implementation. Called by \code{morie_dml_clustered}. See the file header for the source the module follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
.dmlc_multiway_se(infl = data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9)), clusters = c(1, 2, 3, 4, 5, 6, 7, 8), n = 5L)
}
