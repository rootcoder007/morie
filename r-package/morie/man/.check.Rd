% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.check}
\alias{.check}
\title{.check}
\usage{
.check(scores, priority)
}
\arguments{
\item{scores}{Passed to \code{.slvgrf_vec}.}
\item{priority}{Passed to \code{.slvgrf_vec}.}
}
\description{
Part of the slvgrf_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{g}, \code{s}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.check(V, V)
}
