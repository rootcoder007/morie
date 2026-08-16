% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{chronos_mean_scale}
\alias{chronos_mean_scale}
\title{chronos_mean_scale}
\usage{
chronos_mean_scale(x, context = NULL)
}
\arguments{
\item{x}{See Usage.}
\item{context}{Defaults to \code{NULL}.}
}
\description{
Part of the chronos_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{scaled}, \code{scale}, \code{degenerate}, \code{context}, \code{preserves_zero}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
chronos_mean_scale(V)
}
