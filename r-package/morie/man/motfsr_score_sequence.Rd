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
\item{spec}{See Usage.}
\item{sequence}{See Usage.}
\item{alphabet}{See Usage.}
\item{threshold}{Defaults to \code{NULL}.}
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
