% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.airl_key}
\alias{.airl_key}
\title{.airl_key}
\usage{
.airl_key(s)
}
\arguments{
\item{s}{A vector; its length is taken.}
}
\description{
A step of the airl_native implementation. No other function in the package calls it. See the file header for the source the module follows.
}
\value{
One of two values, depending on the branch taken.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.airl_key(V)
}
