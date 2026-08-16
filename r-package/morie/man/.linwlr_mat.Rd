% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.linwlr_mat}
\alias{.linwlr_mat}
\title{.linwlr_mat}
\usage{
.linwlr_mat(x)
}
\arguments{
\item{x}{Optional; may be \code{NULL}. A matrix; passed to \code{as.matrix}.}
}
\description{
Part of the linwlr_native implementation; see the file header for the source it follows.
}
\value{
A matrix, from \code{as.matrix}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.linwlr_mat(V)
}
