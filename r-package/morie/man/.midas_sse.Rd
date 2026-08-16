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
\item{p}{A vector; indexed elementwise.}
\item{X}{A matrix; passed to \code{\%*\%}.}
\item{Y}{Numeric; combined arithmetically in the body.}
\item{K}{Passed to \code{.morie_beta_weights}.}
}
\description{
morie_midas_regression() optimiser closure so the theta-domain guard and the non-finite-SSE guard are directly unit-testable. `X` is the lag-matrix, `Y` the target, `K` the number of high-frequency lags.
}
\value{
One of two values, depending on the branch taken.
}
\keyword{internal}
\examples{
.midas_sse(p = c(1, 2, 3, 4, 5, 6, 7, 8), X = 5L, Y = 5L, K = 5L)
}
