% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{motfsr_alphabet_of}
\alias{motfsr_alphabet_of}
\title{motfsr_alphabet_of}
\usage{
motfsr_alphabet_of(seqs, alphabet) 
}
\arguments{
\item{seqs}{Passed to \code{unlist}.}
\item{alphabet}{Optional; may be \code{NULL}. Coerced to character by the body, with \code{as.character}.}
}
\description{
Part of the motfsr_native implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{sort}.
}
\references{
  Bailey & Elkan (1994) ISMB-94, 28-36.
  Bailey & Elkan (1995) ISMB-95, 21-29.
Base R only.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
motfsr_alphabet_of(V, V)
}
