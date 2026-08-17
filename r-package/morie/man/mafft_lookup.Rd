% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mafft_lookup}
\alias{mafft_lookup}
\title{mafft_lookup}
\usage{
mafft_lookup(M, a, b)
}
\arguments{
\item{M}{A vector; indexed elementwise.}
\item{a}{Passed to \code{paste0}.}
\item{b}{Passed to \code{paste0}.}
}
\description{
Part of the mafft_native implementation; see the file header for the source it follows.
}
\value{
One of two values, depending on the branch taken.
}
\keyword{internal}
\examples{
M <- list(A_A = 4, A_C = -1, C_C = 4)
c(mafft_lookup(M, "A", "A"), mafft_lookup(M, "A", "C"),
  mafft_lookup(M, "G", "T"))
}
