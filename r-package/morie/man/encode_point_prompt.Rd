% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{encode_point_prompt}
\alias{encode_point_prompt}
\title{encode_point_prompt}
\usage{
encode_point_prompt(points, labels, dim = 8, type_embeddings = NULL) 
}
\arguments{
\item{points}{Iterated over elementwise, with \code{lapply}.}
\item{labels}{Coerced to integer by the body, with \code{as.integer}.}
\item{dim}{Passed to \code{.samseg_pos_enc}. Defaults to \code{8}.}
\item{type_embeddings}{Defaults to \code{NULL}.}
}
\description{
Part of the samseg_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{tokens}, \code{n_prompts}, \code{sparse}, \code{note}.
}
\keyword{internal}
\examples{
e <- encode_point_prompt(points = list(c(10, 20), c(30, 40)),
                         labels = c(1, 0), dim = 8)
is.list(e) || is.matrix(e) || is.numeric(e)
}
