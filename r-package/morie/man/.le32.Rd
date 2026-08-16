% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.le32}
\alias{.le32}
\title{Little-endian 32-bit pack of an integer (returns raw bytes)}
\usage{
.le32(n)
}
\arguments{
\item{n}{Coerced to numeric by the body, with \code{as.numeric}.}
}
\description{
Part of the secarg_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{as.raw}.
}
\keyword{internal}
\examples{
.le32(n = 5L)
}
