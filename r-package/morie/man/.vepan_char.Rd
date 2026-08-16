% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.vepan_char}
\alias{.vepan_char}
\title{0-based character access}
\usage{
.vepan_char(s, k)
}
\arguments{
\item{s}{Character; passed to \code{substr}.}
\item{k}{Numeric; combined arithmetically in the body.}
}
\description{
Part of the vepan_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{substr}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.vepan_char(V, V)
}
