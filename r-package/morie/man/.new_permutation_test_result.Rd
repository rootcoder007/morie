% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.new_permutation_test_result}
\alias{.new_permutation_test_result}
\title{.new_permutation_test_result}
\usage{
.new_permutation_test_result(observed_statistic, p_value, null_distribution, n_permutations, alternative, ci_lower = NA_real_, ci_upper = NA_real_)
}
\arguments{
\item{observed_statistic}{See Usage.}
\item{p_value}{See Usage.}
\item{null_distribution}{See Usage.}
\item{n_permutations}{See Usage.}
\item{alternative}{See Usage.}
\item{ci_lower}{Defaults to \code{NA_real_}.}
\item{ci_upper}{Defaults to \code{NA_real_}.}
}
\description{
A step of the bootstrap_methods implementation. Called by \code{paired_permutation_test}, \code{permutation_test}. See the file header for the source the module follows.
}
\value{
The value of \code{structure}.
}
\keyword{internal}
