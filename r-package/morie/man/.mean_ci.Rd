% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.mean_ci}
\alias{.mean_ci}
\title{.mean_ci}
\usage{
.mean_ci(x, confidence = 0.95)
}
\arguments{
\item{x}{A vector; its length is taken.}
\item{confidence}{Numeric; combined arithmetically in the body. Defaults to \code{0.95}.}
}
\description{
Part of the statistics implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{c}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.mean_ci(V)
}
