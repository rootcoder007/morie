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
\item{observed_statistic}{Carried through into a list the body builds.}
\item{p_value}{Carried through into a list the body builds.}
\item{null_distribution}{Carried through into a list the body builds.}
\item{n_permutations}{Carried through into a list the body builds.}
\item{alternative}{Carried through into a list the body builds.}
\item{ci_lower}{Carried through into a list the body builds. Defaults to \code{NA_real_}.}
\item{ci_upper}{Carried through into a list the body builds. Defaults to \code{NA_real_}.}
}
\description{
A step of the bootstrap_methods implementation. Called by \code{paired_permutation_test}, \code{permutation_test}. See the file header for the source the module follows.
}
\value{
The value of \code{structure}.
}
\keyword{internal}
\examples{
.new_permutation_test_result(observed_statistic = c(1, 2, 3, 4, 5, 6, 7, 8), p_value = c(1, 2, 3, 4, 5, 6, 7, 8), null_distribution = c(1, 2, 3, 4, 5, 6, 7, 8), n_permutations = c(1, 2, 3, 4, 5, 6, 7, 8), alternative = c(1, 2, 3, 4, 5, 6, 7, 8))
}
