% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.hrzb2_loss}
\alias{.hrzb2_loss}
\title{Internal: smoothed maximum-score loss. Extracted from the hrzb2()}
\usage{
.hrzb2_loss(b, X, ys, h)
}
\arguments{
\item{b}{Numeric; combined arithmetically in the body.}
\item{X}{A matrix; passed to \code{\%*\%}.}
\item{ys}{Numeric; combined arithmetically in the body.}
\item{h}{Numeric; combined arithmetically in the body.}
}
\description{
optimiser closure so the zero-norm guard is directly unit-testable.
}
\value{
A numeric value.
}
\keyword{internal}
