% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mafft_anchors_from}
\alias{mafft_anchors_from}
\title{mafft_anchors_from}
\usage{
mafft_anchors_from(chain) 
}
\arguments{
\item{chain}{A matrix; indexed by row and column.}
}
\description{
Part of the mafft_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{mapply}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
mafft_anchors_from(V)
}
