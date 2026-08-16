% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.exceptional_law_share}
\alias{.exceptional_law_share}
\title{.exceptional_law_share}
\usage{
.exceptional_law_share(blips, tol = 0.01)
}
\arguments{
\item{blips}{A vector; its length is taken.}
\item{tol}{Passed to \code{<=}. Defaults to \code{0.01}.}
}
\description{
Part of the tmldyn_native implementation; see the file header for the source it follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.exceptional_law_share(V)
}
