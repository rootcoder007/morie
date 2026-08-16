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
\item{scores}{See Usage.}
\item{mean_score}{See Usage.}
\item{se_score}{See Usage.}
\item{ci_lower}{See Usage.}
\item{ci_upper}{See Usage.}
\item{n_folds}{See Usage.}
\item{metric}{See Usage.}
\item{fold_sizes}{See Usage.}
}
\description{
A step of the bootstrap_methods implementation. Called by \code{.boot_cross_validate}, \code{repeated_cv}. See the file header for the source the module follows.
}
\value{
The value of \code{structure}.
}
\keyword{internal}
