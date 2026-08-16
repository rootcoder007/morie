% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.ukfF_chol}
\alias{.ukfF_chol}
\title{.ukfF_chol}
\usage{
.ukfF_chol(a)
}
\arguments{
\item{a}{A matrix; indexed by row and column.}
}
\description{
Part of the ukfF_mixedcase_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{l}, as built in the body.
}
\keyword{internal}
\examples{
M <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
.ukfF_chol(M)
}
