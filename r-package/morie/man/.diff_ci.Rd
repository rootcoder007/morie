% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.diff_ci}
\alias{.diff_ci}
\title{.diff_ci}
\usage{
.diff_ci(x, y, confidence = 0.95, equal_var = TRUE)
}
\arguments{
\item{x}{A vector; its length is taken.}
\item{y}{A vector; its length is taken.}
\item{confidence}{Numeric; combined arithmetically in the body. Defaults to \code{0.95}.}
\item{equal_var}{A flag; the body branches on it. Defaults to \code{TRUE}.}
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
.diff_ci(V, V)
}
