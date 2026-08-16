% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.vepan_key_less}
\alias{.vepan_key_less}
\title{Lexicographic comparison of two pick keys}
\usage{
.vepan_key_less(k1, k2)
}
\arguments{
\item{k1}{A vector; its length is taken and its elements indexed.}
\item{k2}{A vector; indexed elementwise.}
}
\description{
Part of the vepan_native implementation; see the file header for the source it follows.
}
\value{
A logical value.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.vepan_key_less(V, V)
}
