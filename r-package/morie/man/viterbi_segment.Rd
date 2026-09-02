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
\arguments{
\item{text}{Argument \code{text}; see Usage.}
\item{piece_logp}{Argument \code{piece_logp}; see Usage.}
\item{add_prefix}{Argument \code{add_prefix}; see Usage.}
}
\value{
A list with \code{pieces}, \code{logp}, \code{n_pieces}, \code{algorithm}.
}
\keyword{internal}
