% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{train_bpe}
\alias{train_bpe}
\title{Train a BPE model: merge the most frequent adjacent pair greedily}
\usage{
train_bpe(corpus, vocab_size, add_prefix = TRUE)
}
\description{
Train a BPE model: merge the most frequent adjacent pair greedily
}
\value{
A list with \code{merges}, \code{vocab}, \code{vocab_size}, \code{requested}, \code{algorithm}, \code{note}.
}
\keyword{internal}
