% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.new_jackknife_result}
\alias{.new_jackknife_result}
\title{.new_jackknife_result}
\usage{
.new_jackknife_result(estimate, se, ci_lower, ci_upper, bias, n, jackknife_estimates, pseudovalues, influence_values)
}
\arguments{
\item{estimate}{See Usage.}
\item{se}{See Usage.}
\item{ci_lower}{See Usage.}
\item{ci_upper}{See Usage.}
\item{bias}{See Usage.}
\item{n}{See Usage.}
\item{jackknife_estimates}{See Usage.}
\item{pseudovalues}{See Usage.}
\item{influence_values}{See Usage.}
}
\description{
A step of the bootstrap_methods implementation. Called by \code{delete_d_jackknife}, \code{jackknife}. See the file header for the source the module follows.
}
\value{
The value of \code{structure}.
}
\keyword{internal}
