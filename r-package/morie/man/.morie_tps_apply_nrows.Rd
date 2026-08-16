% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_tps_apply_nrows}
\alias{.morie_tps_apply_nrows}
\title{.morie_tps_apply_nrows}
\usage{
.morie_tps_apply_nrows(df, nrows)
}
\arguments{
\item{df}{A matrix; indexed by row and column.}
\item{nrows}{Optional; may be \code{NULL}. Coerced to integer by the body, with \code{as.integer}.}
}
\description{
Part of the tps_io implementation; see the file header for the source it follows.
}
\value{
The value of \code{[}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
M <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
.morie_tps_apply_nrows(M, V)
}
