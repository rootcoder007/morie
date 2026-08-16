% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.wrd2v_softmax}
\alias{.wrd2v_softmax}
\title{.wrd2v_softmax}
\usage{
.wrd2v_softmax(v)
}
\arguments{
\item{v}{Numeric; passed to \code{max}.}
}
\description{
Part of the wrd2v_native implementation; see the file header for the source it follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.wrd2v_softmax(V)
}
