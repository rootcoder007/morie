% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.sv_nanmean_col}
\alias{.sv_nanmean_col}
\title{.sv_nanmean_col}
\usage{
.sv_nanmean_col(M)
}
\arguments{
\item{M}{See Usage.}
}
\description{
Part of the spatial_voting implementation; see the file header for the source it follows.
}
\value{
The value of \code{apply}.
}
\keyword{internal}
\examples{
M <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
.sv_nanmean_col(M)
}
