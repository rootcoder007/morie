% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.wrd2v_scores}
\alias{.wrd2v_scores}
\title{.wrd2v_scores}
\usage{
.wrd2v_scores(O, h)
}
\arguments{
\item{O}{A matrix; passed to \code{\%*\%}.}
\item{h}{A matrix; passed to \code{\%*\%}.}
}
\description{
Part of the wrd2v_native implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{as.numeric}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.wrd2v_scores(V, V)
}
