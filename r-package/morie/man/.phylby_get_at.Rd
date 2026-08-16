% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.phylby_get_at}
\alias{.phylby_get_at}
\title{.phylby_get_at}
\usage{
.phylby_get_at(node, path)
}
\arguments{
\item{node}{A vector; indexed elementwise.}
\item{path}{See Usage.}
}
\description{
Part of the phylby_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{node}, as built in the body.
}
\keyword{internal}
\examples{
.phylby_get_at(node = c(1, 2, 3, 4, 5, 6, 7, 8), path = 5L)
}
