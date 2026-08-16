% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.sasrec_vec}
\alias{.sasrec_vec}
\title{.sasrec_vec}
\usage{
.sasrec_vec(x)
}
\arguments{
\item{x}{A matrix; indexed by row and column.}
}
\description{
Part of the sasRec_native implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{as.numeric}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.sasrec_vec(V)
}
