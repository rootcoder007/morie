% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.gp_alpha_terms}
\alias{.gp_alpha_terms}
\title{.gp_alpha_terms}
\usage{
.gp_alpha_terms(log_prior, theta, theta_p, ll, ll_p, log_q, log_q_p)
}
\arguments{
\item{log_prior}{See Usage.}
\item{theta}{See Usage.}
\item{theta_p}{See Usage.}
\item{ll}{Numeric; combined arithmetically in the body.}
\item{ll_p}{Numeric; combined arithmetically in the body.}
\item{log_q}{Numeric; combined arithmetically in the body.}
\item{log_q_p}{Numeric; combined arithmetically in the body.}
}
\description{
A step of the abcgp_native implementation. Called by \code{.gp_mw_sampler}. See the file header for the source the module follows.
}
\value{
A numeric value.
}
\keyword{internal}
