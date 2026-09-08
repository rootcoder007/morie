% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{arrange_segments}
\alias{arrange_segments}
\title{arrange_segments}
\usage{
arrange_segments(segments) 
}
\arguments{
\item{segments}{A matrix; indexed by row and column.}
}
\description{
Part of the mafft_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{do.call}.
}
\keyword{internal}
\examples{
segs <- rbind(c(0, 0, 4, 5, 0), c(5, 6, 3, 3, 0), c(2, 1, 2, 4, 0))
arrange_segments(segs)
}
