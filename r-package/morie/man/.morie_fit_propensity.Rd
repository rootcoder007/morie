% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_fit_propensity}
\alias{.morie_fit_propensity}
\title{Internal: fit a logistic propensity model and clip to [0.01, 0.99]}
\usage{
.morie_fit_propensity(data, treatment, covariates)
}
\arguments{
\item{data}{See Usage.}
\item{treatment}{Passed to \code{paste}.}
\item{covariates}{Passed to \code{paste}.}
}
\description{
A step of the investigation implementation. Called by \code{morie_run_treatment_effects_analysis}. See the file header for the source the module follows.
}
\value{
The value of \code{pmin}.
}
\keyword{internal}
