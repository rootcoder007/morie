% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.word_vector}
\alias{.word_vector}
\title{.word_vector}
\usage{
.word_vector(word, Z, gram_index, n_min = 3L, n_max = 6L, boundary = TRUE, whole_word = TRUE, hash_buckets = NULL)
}
\arguments{
\item{word}{See Usage.}
\item{Z}{See Usage.}
\item{gram_index}{See Usage.}
\item{n_min}{Defaults to \code{3L}.}
\item{n_max}{Defaults to \code{6L}.}
\item{boundary}{Defaults to \code{TRUE}.}
\item{whole_word}{Defaults to \code{TRUE}.}
\item{hash_buckets}{Defaults to \code{NULL}.}
}
\description{
Part of the fastxt_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{v}, \code{hit}.
}
\references{
Bojanowski, P., Grave, E., Joulin, A. & Mikolov, T. (2017) "Enriching
word vectors with subword information", Transactions of the Association
for Computational Linguistics 5, 135-146, arXiv:1607.04606. Sec. 3.2
for the subword model and the *where* example, Sec. 3.1 for the
objective.
Mikolov, T., Sutskever, I., Chen, K., Corrado, G. & Dean, J. (2013)
"Distributed representations of words and phrases and their
compositionality", NIPS 26, 3111-3119 -- the skipgram with negative
sampling this extends.
}
\keyword{internal}
