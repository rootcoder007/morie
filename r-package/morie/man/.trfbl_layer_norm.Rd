% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.trfbl_layer_norm}
\alias{.trfbl_layer_norm}
\title{.trfbl_layer_norm}
\usage{
.trfbl_layer_norm(x, eps = 1e-05)
}
\arguments{
\item{x}{Passed to \code{rowMeans}.}
\item{eps}{Numeric; combined arithmetically in the body. Defaults to \code{1e-05}.}
}
\description{
Part of the trfbl implementation; see the file header for the source it follows.
}
\value{
The value of \code{sweep}.
}
\keyword{internal}
\examples{
M <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
.trfbl_layer_norm(M)
}
