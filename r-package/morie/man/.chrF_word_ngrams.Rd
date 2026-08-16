% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.chrF_word_ngrams}
\alias{.chrF_word_ngrams}
\title{.chrF_word_ngrams}
\usage{
.chrF_word_ngrams(ws, n)
}
\arguments{
\item{ws}{A vector; its length is taken and its elements indexed.}
\item{n}{Numeric; combined arithmetically in the body.}
}
\description{
A step of the chrF_native implementation. Called by \code{chrf_score}. See the file header for the source the module follows.
}
\value{
The value of \code{lapply}.
}
\keyword{internal}
\examples{
.chrF_word_ngrams(ws = c(1, 2, 3, 4, 5, 6, 7, 8), n = 5L)
}
