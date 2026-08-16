% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{residue_vectors}
\alias{residue_vectors}
\title{residue_vectors}
\usage{
residue_vectors(group, weights = NULL, seq_type = "aa")
}
\arguments{
\item{group}{The body requires: mafft: sequences in a group must be aligned to the same length.}
\item{weights}{Optional; may be \code{NULL}. A vector; its length is taken and its elements indexed.}
\item{seq_type}{Compared against \code{"nt"}. Defaults to \code{"aa"}.}
}
\description{
Part of the mafft_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{list}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
residue_vectors(V)
}
