% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{fasttext}
\alias{fasttext}
\title{fasttext}
\usage{
fasttext(corpus, dim = 50L, n_min = 3L, n_max = 6L, window = 5L, epochs = 5L, lr = 0.05, negative = 5L, min_count = 1L, boundary = TRUE, whole_word = TRUE, hash_buckets = NULL, seed = 0L)
}
\arguments{
\item{corpus}{See Usage.}
\item{dim}{Defaults to \code{50L}.}
\item{n_min}{Defaults to \code{3L}.}
\item{n_max}{Defaults to \code{6L}.}
\item{window}{Defaults to \code{5L}.}
\item{epochs}{Defaults to \code{5L}.}
\item{lr}{Defaults to \code{0.05}.}
\item{negative}{Defaults to \code{5L}.}
\item{min_count}{Defaults to \code{1L}.}
\item{boundary}{Defaults to \code{TRUE}.}
\item{whole_word}{Defaults to \code{TRUE}.}
\item{hash_buckets}{Defaults to \code{NULL}.}
\item{seed}{Defaults to \code{0L}.}
}
\description{
Part of the fastxt_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{vectors}, \code{vocab}, \code{index}, \code{ngrams}, \code{ngram_index}, \code{Z}, \code{context}, \code{loss_history}, \code{final_loss}, \code{oov}, \code{n_vocab}, \code{n_ngrams}, \code{dim}, \code{n_min}, \code{n_max}, \code{hash_buckets}, \code{method}.
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
