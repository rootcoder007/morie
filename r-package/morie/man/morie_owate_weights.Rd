% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_owate_weights}
\alias{morie_owate_weights}
\title{Theorem 5.4 / Corollary 5.2 OWATE weights}
\usage{
morie_owate_weights(pscore, sigma2_treated = NULL, sigma2_control = NULL) 
}
\arguments{
\item{pscore}{Numeric propensity scores in (0, 1).}
\item{sigma2_treated,sigma2_control}{Optional conditional variances.}
}
\description{
Theorem 5.4 / Corollary 5.2 OWATE weights
}
\value{
A numeric vector of weights, one per observation.
}
\keyword{internal}
