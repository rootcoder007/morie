% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_variable_importance}
\alias{morie_variable_importance}
\title{Rank the columns of X by data-adaptive importance}
\usage{
morie_variable_importance(y, X, candidate_strata = NULL, method = "cv-tmle", n_folds = 10, names = NULL, ...)
}
\arguments{
\item{y}{Numeric outcome.}
\item{X}{Numeric matrix of predictors.}
\item{candidate_strata}{Optional levels to search over.}
\item{method}{One of the three honest methods.}
\item{n_folds}{Number of V-fold splits.}
\item{names}{Optional variable names; defaults to \code{"X1"},   \code{"X2"}, ...}
\item{...}{Forwarded to \code{morie_tmle_data_adaptive}.}
}
\description{
For each column the contrast is estimated with that column as the exposure and everything else as the covariate, and the results are sorted by absolute estimate.
}
\value{
A list of per-variable results sorted by absolute estimate.
}
\keyword{internal}
