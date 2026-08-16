% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.hrzi1_obj}
\alias{.hrzi1_obj}
\title{Internal: Ichimura single-index leave-one-out objective. Extracted}
\usage{
.hrzi1_obj(b, X, y, h0)
}
\arguments{
\item{b}{Numeric; combined arithmetically in the body.}
\item{X}{A matrix; passed to \code{\%*\%}.}
\item{y}{A matrix; passed to \code{\%*\%}.}
\item{h0}{Numeric; combined arithmetically in the body.}
}
\description{
from the hrzi1() optimiser closure so the zero-norm guard is directly unit-testable. `h0` is the kernel bandwidth.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
.hrzi1_obj(b = c(1, 2, 3, 4, 5, 6, 7, 8), X = 5L, y = c(1, 2, 3, 4, 5, 6, 7, 8), h0 = c(1, 2, 3, 4, 5, 6, 7, 8))
}
