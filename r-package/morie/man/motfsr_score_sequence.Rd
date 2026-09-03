% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{motfsr_score_sequence}
\alias{motfsr_score_sequence}
\title{motfsr_score_sequence}
\usage{
motfsr_score_sequence(spec, sequence, alphabet, threshold = NULL) 
}
\arguments{
\item{spec}{A vector; its length is taken and its elements indexed.}
\item{sequence}{Coerced to character by the body, with \code{as.character}.}
\item{alphabet}{A vector; its length is taken.}
\item{threshold}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
}
\description{
Part of the motfsr_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{scores}, \code{hits}.
}
\references{
  Bailey & Elkan (1994) ISMB-94, 28-36.
  Bailey & Elkan (1995) ISMB-95, 21-29.
Base R only.
}
\keyword{internal}
\examples{
motfsr_score_sequence(spec = data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9)), sequence = 5L,
  alphabet = c(1, 2, 3, 4, 5, 6, 7, 8))
}
