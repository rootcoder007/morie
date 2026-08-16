% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.smixd_negreml}
\alias{.smixd_negreml}
\title{Internal: spatial-mixed-model REML negative log-likelihood}
\usage{
.smixd_negreml(theta, D, n, X, y, p)
}
\arguments{
\item{theta}{See Usage.}
\item{D}{See Usage.}
\item{n}{See Usage.}
\item{X}{See Usage.}
\item{y}{See Usage.}
\item{p}{See Usage.}
}
\description{
Extracted from the smixd() optimiser closure so the non-positive- -definite-covariance, singular-information and non-positive-variance guards are all directly unit-testable. `theta` = c(log phi, log nu).
}
\value{
A numeric value.
}
\keyword{internal}
