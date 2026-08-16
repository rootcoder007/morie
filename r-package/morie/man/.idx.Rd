% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.idx}
\alias{.idx}
\title{Helper: subset rows of a vector or matrix}
\usage{
.idx(data, idx)
}
\arguments{
\item{data}{A matrix; indexed by row and column.}
\item{idx}{See Usage.}
}
\description{
A step of the bootstrap_methods implementation. Called by \code{.bca_interval}, \code{block_bootstrap}, \code{bootstrap} and 3 others in the module. See the file header for the source the module follows.
}
\value{
One of two values, depending on the branch taken.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.idx(V, V)
}
