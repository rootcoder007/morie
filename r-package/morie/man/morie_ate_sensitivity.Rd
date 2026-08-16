% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_ate_sensitivity}
\alias{morie_ate_sensitivity}
\title{l1 sensitivity of a TMLE of the ATE}
\usage{
morie_ate_sensitivity(n, g_min, y_range = 1)
}
\arguments{
\item{n}{Number of observations.}
\item{g_min}{Propensity truncation bound.}
\item{y_range}{Range of the outcome.}
}
\description{
The bound carries :math:`1/g_{\min}`, so it is :math:`2R/(n g_{\min})` and NOT :math:`O(1/n)`. Truncating the propensity score is part of the privacy guarantee.
}
\value{
A list with the sensitivity and a few related quantities.
}
\keyword{internal}
