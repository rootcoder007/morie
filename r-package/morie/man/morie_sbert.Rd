% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_sbert}
\alias{morie_sbert}
\title{morie_sbert}
\usage{
morie_sbert(pairs, embed)
}
\arguments{
\item{pairs}{See Usage.}
\item{embed}{See Usage.}
}
\description{
Part of the sbert_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{sts_score}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
D <- data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9))
morie_sbert(D, V)
}
