% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.gram_slot}
\alias{.gram_slot}
\title{.gram_slot}
\usage{
.gram_slot(g, gram_index, hash_buckets)
}
\arguments{
\item{g}{See Usage.}
\item{gram_index}{See Usage.}
\item{hash_buckets}{See Usage.}
}
\description{
Part of the fastxt_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{gi}, as built in the body.
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
