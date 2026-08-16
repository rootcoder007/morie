% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.subwords}
\alias{.subwords}
\title{.subwords}
\usage{
.subwords(word, n_min = 3L, n_max = 6L, boundary = TRUE, whole_word = TRUE)
}
\arguments{
\item{word}{Coerced to character by the body, with \code{as.character}.}
\item{n_min}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{3L}.}
\item{n_max}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{6L}.}
\item{boundary}{A flag; the body branches on it. Defaults to \code{TRUE}.}
\item{whole_word}{A flag; the body branches on it. Defaults to \code{TRUE}.}
}
\description{
A step of the fastxt_native implementation. Called by \code{.word_vector}, \code{fasttext}. See the file header for the source the module follows.
}
\value{
The value of \code{grams}, as built in the body.
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
\examples{
.subwords(word = 5L)
}
