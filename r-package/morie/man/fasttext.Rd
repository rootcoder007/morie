% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{fasttext}
\alias{fasttext}
\title{fasttext}
\usage{
fasttext(corpus, dim = 50L, n_min = 3L, n_max = 6L, window = 5L, epochs = 5L, 
    lr = 0.05, negative = 5L, min_count = 1L, boundary = TRUE, whole_word = TRUE, 
    hash_buckets = NULL, seed = 0L) 
}
\arguments{
\item{corpus}{Passed to \code{.as_docs}.}
\item{dim}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{50L}.}
\item{n_min}{Passed to \code{.subwords}. Defaults to \code{3L}.}
\item{n_max}{Passed to \code{.subwords}. Defaults to \code{6L}.}
\item{window}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{5L}.}
\item{epochs}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{5L}.}
\item{lr}{Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{0.05}.}
\item{negative}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{5L}.}
\item{min_count}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{1L}.}
\item{boundary}{Passed to \code{.subwords}. Defaults to \code{TRUE}.}
\item{whole_word}{Passed to \code{.subwords}. Defaults to \code{TRUE}.}
\item{hash_buckets}{Optional; may be \code{NULL}. Passed to \code{.gram_slot}.}
\item{seed}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{0L}.}
}
\description{
A step of the fastxt_native implementation. Called by \code{morie_fastxt}. See the file header for the source the module follows.
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
