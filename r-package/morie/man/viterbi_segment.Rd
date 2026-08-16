% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{viterbi_segment}
\alias{viterbi_segment}
\title{Unigram LM segmentation by Viterbi over the split lattice}
\usage{
viterbi_segment(text, piece_logp, add_prefix = TRUE)
}
\description{
Unigram LM segmentation by Viterbi over the split lattice
}
\value{
A list with \code{pieces}, \code{logp}, \code{n_pieces}, \code{algorithm}.
}
\keyword{internal}
