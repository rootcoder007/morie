% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.smoopt_K}
\alias{.smoopt_K}
\title{Vectorise the (n, n) gram matrix K into a column-stored numeric}
\usage{
.smoopt_K(K)
}
\arguments{
\item{K}{A vector; its length is taken and its elements indexed.}
}
\description{
matrix if it isn\\'t already.
}
\value{
The value of \code{M}, as built in the body.
}
\keyword{internal}
\examples{
M <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
.smoopt_K(M)
}
