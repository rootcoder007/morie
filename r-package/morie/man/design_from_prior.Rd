% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{design_from_prior}
\alias{design_from_prior}
\title{Sobol design pushed through the prior\'s inverse CDF}
\usage{
design_from_prior(n, prior_ppf, dim = NULL, skip = 1L)
}
\arguments{
\item{n}{Number of points.}
\item{prior_ppf}{Either a list of per-parameter quantile functions   or a list \code{list(lo, hi)} for independent uniform priors.}
\item{dim}{Optional dimension check.}
\item{skip}{Drop that many leading Sobol points (default 1, so the   origin is not sent to the prior corner).}
}
\description{
Sobol design pushed through the prior\'s inverse CDF
}
\value{
Numeric matrix with \code{n} rows.
}
\keyword{internal}
