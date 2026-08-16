% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.motiff_triads}
\alias{.motiff_triads}
\title{.motiff_triads}
\usage{
.motiff_triads(adj, n)
}
\arguments{
\item{adj}{A matrix; indexed by row and column.}
\item{n}{A count; the body uses it as \code{seq_len(...)}.}
}
\description{
Part of the motiff_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{ffl}, \code{cycle3}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.motiff_triads(V, V)
}
