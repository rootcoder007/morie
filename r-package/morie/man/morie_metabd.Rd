% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_metabd}
\alias{morie_metabd}
\title{morie_metabd}
\usage{
morie_metabd(tnfs, coverages = NULL, lengths = NULL, threshold = 0.15, min_bin_size = 2e+05)
}
\arguments{
\item{tnfs}{See Usage.}
\item{coverages}{Defaults to \code{NULL}.}
\item{lengths}{Defaults to \code{NULL}.}
\item{threshold}{Defaults to \code{0.15}.}
\item{min_bin_size}{Defaults to \code{2e+05}.}
}
\description{
Part of the metabd_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{bin_contigs}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
morie_metabd(V)
}
