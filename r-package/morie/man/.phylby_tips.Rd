% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.phylby_tips}
\alias{.phylby_tips}
\title{.phylby_tips}
\usage{
.phylby_tips(node, out = NULL)
}
\arguments{
\item{node}{A vector; its length is taken and its elements indexed.}
\item{out}{Optional; may be \code{NULL}. Passed to \code{.phylby_tips}.}
}
\description{
Part of the phylby_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.phylby_tips(V)
}
