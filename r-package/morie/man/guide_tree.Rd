% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{guide_tree}
\alias{guide_tree}
\title{guide_tree}
\usage{
guide_tree(D)
}
\arguments{
\item{D}{A matrix; indexed by row and column.}
}
\description{
Part of the mafft_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{merges}, as built in the body.
}
\keyword{internal}
\examples{
seqs <- c("MKVLA", "MKVLG", "MRVLA")
guide_tree(sixtuple_distance(seqs))
}
