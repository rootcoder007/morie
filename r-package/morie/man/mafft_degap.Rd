% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mafft_degap}
\alias{mafft_degap}
\title{mafft_degap}
\usage{
mafft_degap(group) 
}
\arguments{
\item{group}{A vector; its length is taken and its elements indexed.}
}
\description{
Part of the mafft_native implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{vapply}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
mafft_degap(V)
}
