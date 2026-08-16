% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.midas_sse}
\alias{.midas_sse}
\title{Internal: MIDAS sum-of-squared-errors objective. Extracted from the}
\usage{
.midas_sse(p, X, Y, K)
}
\arguments{
\item{p}{See Usage.}
\item{X}{See Usage.}
\item{Y}{See Usage.}
\item{K}{See Usage.}
}
\description{
morie_midas_regression() optimiser closure so the theta-domain guard and the non-finite-SSE guard are directly unit-testable. `X` is the lag-matrix, `Y` the target, `K` the number of high-frequency lags.
}
\value{
One of two values, depending on the branch taken.
}
\keyword{internal}
