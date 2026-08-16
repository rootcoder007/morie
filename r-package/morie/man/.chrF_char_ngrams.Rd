% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.chrF_char_ngrams}
\alias{.chrF_char_ngrams}
\title{.chrF_char_ngrams}
\usage{
.chrF_char_ngrams(s, n)
}
\arguments{
\item{s}{A vector; its length is taken.}
\item{n}{Numeric; combined arithmetically in the body.}
}
\description{
A step of the chrF_native implementation. Called by \code{chrf_score}. See the file header for the source the module follows.
}
\value{
The value of \code{substring}.
}
\keyword{internal}
\examples{
.chrF_char_ngrams(s = 5L, n = 5L)
}
