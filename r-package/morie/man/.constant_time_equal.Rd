% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.constant_time_equal}
\alias{.constant_time_equal}
\title{.constant_time_equal}
\usage{
.constant_time_equal(a, b)
}
\arguments{
\item{a}{A vector; its length is taken.}
\item{b}{A vector; its length is taken.}
}
\description{
Part of the sechsh_native implementation; see the file header for the source it follows.
}
\value{
A logical value.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.constant_time_equal(V, V)
}
