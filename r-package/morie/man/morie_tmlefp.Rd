% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_tmlefp}
\alias{morie_tmlefp}
\title{Optimal-overlap estimands and their trimming rules}
\usage{
morie_tmlefp(y, treatment, pscore, sigma2_treated = NULL, sigma2_control = NULL, estimand = "ate")
}
\arguments{
\item{y}{Numeric outcome vector.}
\item{treatment}{Integer 0/1 treatment indicators.}
\item{pscore}{Numeric propensity scores in (0, 1).}
\item{sigma2_treated,sigma2_control}{Optional conditional variances.}
\item{estimand}{One of \code{"ate"} or \code{"att"}.}
}
\description{
Returns the OSATE (Theorem 5.2 or 5.3), the untrimmed full-sample effect, the OWATE (Theorem 5.4), and the variance bound from Theorem 6.1.
}
\value{
A list with the OSATE, full-sample, and OWATE estimates, the   trim rule, and the variance bound.
}
\keyword{internal}
