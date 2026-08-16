% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.as_docs}
\alias{.as_docs}
\title{.as_docs}
\usage{
.as_docs(corpus)
}
\arguments{
\item{corpus}{The body requires: fasttext: corpus must not be None.}
}
\description{
A step of the fastxt_native implementation. Called by \code{fasttext}. See the file header for the source the module follows.
}
\value{
The value of \code{docs}, as built in the body.
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
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.as_docs(V)
}
