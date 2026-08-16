% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.scintg_kmeans_init}
\alias{.scintg_kmeans_init}
\title{.scintg_kmeans_init}
\usage{
.scintg_kmeans_init(Zn, K, seed)
}
\arguments{
\item{Zn}{A matrix; indexed by row and column.}
\item{K}{A count; the body uses it as \code{seq_len(...)}.}
\item{seed}{See Usage.}
}
\description{
Part of the scintg_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{centres}, as built in the body.
}
\keyword{internal}
