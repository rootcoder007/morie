% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_tmle_diff_kernel}
\alias{morie_tmle_diff_kernel}
\title{Differentially private TMLE of the ATE}
\usage{
morie_tmle_diff_kernel(y, D, X, epsilon = 1, g_min = 0.05, seed = 0, g = NULL, Q1 = NULL, 
    Q0 = NULL) 
}
\arguments{
\item{y,D,X}{Outcome, treatment, covariates.}
\item{epsilon}{Privacy parameter.}
\item{g_min}{Propensity truncation.}
\item{seed}{Seed for the shared generator.}
\item{g,Q1,Q0}{Optional pre-fitted nuisances.}
}
\description{
The propensity score is truncated at \code{g_min} -- which bounds the sensitivity and is therefore part of the privacy guarantee, not a numerical convenience.
}
\value{
A list with the private estimate, the non-private TMLE for   comparison, the sensitivity, and the private interval.
}
\keyword{internal}
\examples{
set.seed(1)
n <- 120
X <- matrix(rnorm(n * 2), n, 2)
D <- rbinom(n, 1, plogis(0.4 * X[, 1]))
y <- plogis(0.5 * D + X[, 1] + rnorm(n, 0, 0.3))
r <- morie_tmle_diff_kernel(y, D, X, epsilon = 1, g_min = 0.05, seed = 1)
is.list(r)
}
