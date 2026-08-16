% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.mor_cp_segmeans}
\alias{.mor_cp_segmeans}
\title{Segment means of the 0-based bounds vector}
\usage{
.mor_cp_segmeans(x, bounds)
}
\arguments{
\item{x}{A vector; indexed elementwise.}
\item{bounds}{A vector; its length is taken and its elements indexed.}
}
\description{
Part of the pelt_native implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{vapply}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.mor_cp_segmeans(V, V)
}
