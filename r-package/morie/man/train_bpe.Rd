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
\arguments{
\item{corpus}{Argument \code{corpus}; see Usage.}
\item{vocab_size}{Argument \code{vocab_size}; see Usage.}
\item{add_prefix}{Argument \code{add_prefix}; see Usage.}
}
\value{
A list with \code{merges}, \code{vocab}, \code{vocab_size}, \code{requested}, \code{algorithm}, \code{note}.
}
\keyword{internal}
\examples{
train_bpe(corpus = c(1, 2, 3, 4, 5, 6, 7, 8), vocab_size = 5L)
}
