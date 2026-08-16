% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{sts_score}
\alias{sts_score}
\title{sts_score}
\usage{
sts_score(pairs, embed)
}
\arguments{
\item{pairs}{A vector; its length is taken and its elements indexed.}
\item{embed}{See Usage.}
}
\description{
Part of the sbert_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{scores}, \code{embed_calls}, \code{n_pairs}, \code{cross_encoder_calls}, \code{method}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
D <- data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9))
sts_score(D, V)
}
