% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{start_allowed}
\alias{start_allowed}
\title{start_allowed}
\usage{
start_allowed(labels)
}
\arguments{
\item{labels}{Iterated over elementwise, with \code{sapply}.}
}
\description{
A step of the benRea_native implementation. Called by \code{viterbi_decode}. See the file header for the source the module follows.
}
\value{
A vector, from \code{sapply}.
}
\keyword{internal}
\examples{
S <- c("a", "b", "c")
start_allowed(S)
}
