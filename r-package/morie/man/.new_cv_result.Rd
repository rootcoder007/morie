% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.new_cv_result}
\alias{.new_cv_result}
\title{.new_cv_result}
\usage{
.new_cv_result(scores, mean_score, se_score, ci_lower, ci_upper, n_folds, metric, fold_sizes)
}
\arguments{
\item{scores}{Carried through into a list the body builds.}
\item{mean_score}{Carried through into a list the body builds.}
\item{se_score}{Carried through into a list the body builds.}
\item{ci_lower}{Carried through into a list the body builds.}
\item{ci_upper}{Carried through into a list the body builds.}
\item{n_folds}{Carried through into a list the body builds.}
\item{metric}{Carried through into a list the body builds.}
\item{fold_sizes}{Carried through into a list the body builds.}
}
\description{
A step of the bootstrap_methods implementation. Called by \code{.boot_cross_validate}, \code{repeated_cv}. See the file header for the source the module follows.
}
\value{
The value of \code{structure}.
}
\keyword{internal}
