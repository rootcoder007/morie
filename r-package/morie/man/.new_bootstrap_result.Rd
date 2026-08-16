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
\item{estimate}{Carried through into a list the body builds.}
\item{se}{Carried through into a list the body builds.}
\item{ci_lower}{Carried through into a list the body builds.}
\item{ci_upper}{Carried through into a list the body builds.}
\item{bias}{Carried through into a list the body builds.}
\item{n_boot}{Carried through into a list the body builds.}
\item{method}{Carried through into a list the body builds.}
\item{ci_method}{Carried through into a list the body builds.}
\item{boot_distribution}{Carried through into a list the body builds.}
\item{original_estimate}{Carried through into a list the body builds.}
\item{acceleration}{Carried through into a list the body builds. Defaults to \code{0}.}
}
\description{
A step of the bootstrap_methods implementation. Called by \code{block_bootstrap}, \code{bootstrap}, \code{parametric_bootstrap} and 2 others in the module. See the file header for the source the module follows.
}
\value{
The value of \code{structure}.
}
\keyword{internal}
