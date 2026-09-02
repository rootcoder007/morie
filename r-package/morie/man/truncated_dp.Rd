% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{truncated_dp}
\alias{truncated_dp}
\title{Truncated Dirichlet process draw}
\usage{
truncated_dp(alpha, K, base_sampler = NULL, seed = 0, renormalise = TRUE) 
}
\description{
Truncated Dirichlet process draw
}
\arguments{
\item{alpha}{Argument \code{alpha}; see Usage.}
\item{K}{Argument \code{K}; see Usage.}
\item{base_sampler}{Argument \code{base_sampler}; see Usage.}
\item{seed}{Argument \code{seed}; see Usage.}
\item{renormalise}{Argument \code{renormalise}; see Usage.}
}
\value{
A list with \code{estimate}, \code{weights}, \code{atoms}, \code{discarded_mass}, \code{expected_discarded}, \code{renormalised}, \code{K}, \code{alpha}, \code{method}, \code{note}.
}
\keyword{internal}
\examples{
truncated_dp(alpha = 0.5, K = 5L)
}
