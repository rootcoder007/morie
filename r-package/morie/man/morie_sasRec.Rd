% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_sasRec}
\alias{morie_sasRec}
\title{morie_sasRec}
\usage{
morie_sasRec(E, WQ, WK, WV, mask = NULL)
}
\arguments{
\item{E}{See Usage.}
\item{WQ}{See Usage.}
\item{WK}{See Usage.}
\item{WV}{See Usage.}
\item{mask}{Defaults to \code{NULL}.}
}
\description{
Part of the sasRec_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{self_attention}.
}
\keyword{internal}
\examples{
morie_sasRec(E = c(1, 2, 3, 4, 5, 6, 7, 8), WQ = 5L, WK = 5L, WV = c(1, 2, 3, 4, 5, 6, 7, 8))
}
