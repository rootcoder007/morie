% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.new_bootstrap_result}
\alias{.new_bootstrap_result}
\title{.new_bootstrap_result}
\usage{
.new_bootstrap_result(estimate, se, ci_lower, ci_upper, bias, n_boot, method, ci_method, boot_distribution, original_estimate, acceleration = 0)
}
\arguments{
\item{estimate}{See Usage.}
\item{se}{See Usage.}
\item{ci_lower}{See Usage.}
\item{ci_upper}{See Usage.}
\item{bias}{See Usage.}
\item{n_boot}{See Usage.}
\item{method}{See Usage.}
\item{ci_method}{See Usage.}
\item{boot_distribution}{See Usage.}
\item{original_estimate}{See Usage.}
\item{acceleration}{Defaults to \code{0}.}
}
\description{
A step of the bootstrap_methods implementation. Called by \code{block_bootstrap}, \code{bootstrap}, \code{parametric_bootstrap} and 2 others in the module. See the file header for the source the module follows.
}
\value{
The value of \code{structure}.
}
\keyword{internal}
