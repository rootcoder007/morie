% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mafft_clean}
\alias{mafft_clean}
\title{mafft_clean}
\usage{
mafft_clean(seqs, seq_type = NULL)
}
\arguments{
\item{seqs}{Passed to \code{unlist}.}
\item{seq_type}{Optional; may be \code{NULL}. One of \code{"aa"}, \code{"nt"}.}
}
\description{
Part of the mafft_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{seqs}, \code{seq_type}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
mafft_clean(V)
}
