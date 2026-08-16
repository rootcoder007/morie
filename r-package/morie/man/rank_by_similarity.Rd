% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{rank_by_similarity}
\alias{rank_by_similarity}
\title{rank_by_similarity}
\usage{
rank_by_similarity(query, corpus_embeddings, top_k = 5)
}
\arguments{
\item{query}{Passed to \code{cosine_similarity}.}
\item{corpus_embeddings}{Passed to \code{.sbert_mat}.}
\item{top_k}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{5}.}
}
\description{
Part of the sbert_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{ranking}, \code{n_corpus}, \code{forward_passes}, \code{note}.
}
\keyword{internal}
\examples{
rank_by_similarity(query = 5L, corpus_embeddings = c(1, 2, 3, 4, 5, 6, 7, 8))
}
