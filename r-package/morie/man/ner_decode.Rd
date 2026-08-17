% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{ner_decode}
\alias{ner_decode}
\title{ner_decode}
\usage{
ner_decode(emissions, types, decoder = "viterbi", transition_scores = NULL, gold = NULL)
}
\arguments{
\item{emissions}{A vector; its length is taken and its elements indexed.}
\item{types}{Passed to \code{bio_labels}.}
\item{decoder}{One of \code{"greedy"}, \code{"viterbi"}. Defaults to \code{"viterbi"}.}
\item{transition_scores}{Passed to \code{viterbi_decode}.}
\item{gold}{Optional; may be \code{NULL}. Coerced to list by the body, with \code{as.list}.}
}
\description{
A step of the benRea_native implementation. Called by \code{morie_benRea}. See the file header for the source the module follows.
}
\value{
The value of \code{payload}, as built in the body.
}
\keyword{internal}
\examples{
emissions <- list(c(2, 0, 0), c(0, 2, 0), c(0, 0, 2), c(2, 0, 0))
r <- ner_decode(emissions, types = list("PER"), decoder = "viterbi")
c(r$n_spans, r$valid)
}
