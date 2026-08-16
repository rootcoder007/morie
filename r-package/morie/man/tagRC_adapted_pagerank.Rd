% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{tagRC_adapted_pagerank}
\alias{tagRC_adapted_pagerank}
\title{tagRC_adapted_pagerank}
\usage{
tagRC_adapted_pagerank(adjacency, nodes, p = NULL, d = 0.7, iters = 200, tol = 1e-12)
}
\arguments{
\item{adjacency}{A vector; indexed elementwise.}
\item{nodes}{See Usage.}
\item{p}{Defaults to \code{NULL}.}
\item{d}{Numeric; combined arithmetically in the body. Defaults to \code{0.7}.}
\item{iters}{A count; the body uses it as \code{seq_len(...)}. Defaults to \code{200}.}
\item{tol}{Defaults to \code{1e-12}.}
}
\description{
Part of the tagRC_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{w}, \code{ranking}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
D <- data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9))
tagRC_adapted_pagerank(D, V)
}
