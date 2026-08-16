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
\item{word}{Passed to \code{.subwords}.}
\item{Z}{A matrix; indexed by row and column.}
\item{gram_index}{Passed to \code{.gram_slot}.}
\item{n_min}{Passed to \code{.subwords}. Defaults to \code{3L}.}
\item{n_max}{Passed to \code{.subwords}. Defaults to \code{6L}.}
\item{boundary}{Passed to \code{.subwords}. Defaults to \code{TRUE}.}
\item{whole_word}{Passed to \code{.subwords}. Defaults to \code{TRUE}.}
\item{hash_buckets}{Passed to \code{.gram_slot}.}
}
\description{
A step of the fastxt_native implementation. Called by \code{fasttext}. See the file header for the source the module follows.
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
