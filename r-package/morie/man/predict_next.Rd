% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{predict_next}
\alias{predict_next}
\title{predict_next}
\usage{
predict_next(state, item_embeddings, top_k = 5, exclude = list()) 
}
\arguments{
\item{state}{Passed to \code{.sasrec_vec}.}
\item{item_embeddings}{Passed to \code{.sasrec_mat}.}
\item{top_k}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{5}.}
\item{exclude}{Passed to \code{unlist}. Defaults to \code{list()}.}
}
\description{
Part of the sasRec_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{ranking}, \code{n_scored}, \code{method}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
predict_next(V, V)
}
