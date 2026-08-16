% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{self_attention}
\alias{self_attention}
\title{self_attention}
\usage{
self_attention(E, WQ, WK, WV, mask = NULL)
}
\arguments{
\item{E}{Passed to \code{.sasrec_mat}.}
\item{WQ}{A matrix; passed to \code{nrow}.}
\item{WK}{A matrix; passed to \code{ncol}.}
\item{WV}{A matrix; passed to \code{ncol}.}
\item{mask}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
}
\description{
Part of the sasRec_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{output}, \code{weights}, \code{note}.
}
\keyword{internal}
\examples{
self_attention(E = c(1, 2, 3, 4, 5, 6, 7, 8), WQ = 5L, WK = 5L, WV = c(1, 2, 3, 4, 5, 6, 7, 8))
}
