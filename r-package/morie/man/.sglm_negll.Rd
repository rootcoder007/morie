% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.sglm_negll}
\alias{.sglm_negll}
\title{Internal: spatial-GLM profile negative log-likelihood in log(phi)}
\usage{
.sglm_negll(log_phi, D, n, X, y)
}
\arguments{
\item{log_phi}{Numeric; passed to \code{exp}.}
\item{D}{Numeric; combined arithmetically in the body.}
\item{n}{A matrix; passed to \code{diag}.}
\item{X}{Passed to \code{backsolve}.}
\item{y}{Passed to \code{backsolve}.}
}
\description{
Extracted from the sglm() optimiser closure so the non-positive- -definite-covariance guard is directly unit-testable. `D` is the distance matrix, `n` the sample size, `X`/`y` the design and response.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
.sglm_negll(log_phi = c(1, 2, 3, 4, 5, 6, 7, 8), D = 5L, n = 5L, X = c(1, 2, 3, 4, 5, 6, 7, 8), y = c(1, 2, 3, 4, 5, 6, 7, 8))
}
