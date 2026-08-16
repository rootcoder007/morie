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
\item{estimate}{Carried through into a list the body builds.}
\item{se}{Carried through into a list the body builds.}
\item{ci_lower}{Carried through into a list the body builds.}
\item{ci_upper}{Carried through into a list the body builds.}
\item{bias}{Carried through into a list the body builds.}
\item{n}{Carried through into a list the body builds.}
\item{jackknife_estimates}{Carried through into a list the body builds.}
\item{pseudovalues}{Carried through into a list the body builds.}
\item{influence_values}{Carried through into a list the body builds.}
}
\description{
A step of the bootstrap_methods implementation. Called by \code{delete_d_jackknife}, \code{jackknife}. See the file header for the source the module follows.
}
\value{
The value of \code{structure}.
}
\keyword{internal}
