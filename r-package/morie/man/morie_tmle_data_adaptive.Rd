% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_tmle_data_adaptive}
\alias{morie_tmle_data_adaptive}
\title{Contrast between data-discovered exposure levels, done honestly}
\usage{
morie_tmle_data_adaptive(y, D, X, candidate_strata = NULL, method = "cv-tmle", n_folds = 10, trim = 0.01, ridge = 1e-08, level = 0.95, bounds = NULL)
}
\arguments{
\item{y,D,X}{Outcome, exposure, covariates.}
\item{candidate_strata}{Optional vector of candidate exposure   levels; defaults to the sorted distinct exposures.}
\item{method}{One of \code{"cv-tmle"}, \code{"sample-split"},   \code{"naive"}.}
\item{n_folds}{Number of V-fold splits.}
\item{trim}{Positivity trim for the propensity score.}
\item{ridge}{Optional ridge for the nuisance fits.}
\item{level}{Confidence level for the interval.}
\item{bounds}{Optional \code{c(lower, upper)}; defaults to the   data range.}
}
\description{
Contrast between data-discovered exposure levels, done honestly
}
\value{
A list with \code{estimate}, \code{se}, \code{ci}, the   per-split levels and estimates, and the separation diagnostics.
}
\keyword{internal}
