% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.kvmse_quantize}
\alias{.kvmse_quantize}
\title{.kvmse_quantize}
\usage{
.kvmse_quantize(y, cb)
}
\arguments{
\item{y}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{cb}{See Usage.}
}
\description{
Part of the tqkmse implementation; see the file header for the source it follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.kvmse_quantize(V, V)
}
