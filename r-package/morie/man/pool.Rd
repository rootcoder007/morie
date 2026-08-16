% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{pool}
\alias{pool}
\title{pool}
\usage{
pool(token_vectors, mode = "mean", mask = NULL)
}
\arguments{
\item{token_vectors}{Passed to \code{.sbert_mat}.}
\item{mode}{One of \code{"cls"}, \code{"max"}. Defaults to \code{"mean"}.}
\item{mask}{Defaults to \code{NULL}.}
}
\description{
Part of the sbert_native implementation; see the file header for the source it follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
pool(V)
}
