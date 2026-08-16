% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.samdec_vec}
\alias{.samdec_vec}
\title{.samdec_vec}
\usage{
.samdec_vec(x)
}
\arguments{
\item{x}{A matrix; indexed by row and column.}
}
\description{
Part of the samdec_native implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{as.numeric}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.samdec_vec(V)
}
