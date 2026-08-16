% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.otis_causal_estimate}
\alias{.otis_causal_estimate}
\title{CausalEstimate constructor (R analogue of the python dataclass)}
\usage{
.otis_causal_estimate(estimator, ate, ate_se, ate_pval, n, n_treated, p_treat, notes = list())
}
\arguments{
\item{estimator}{See Usage.}
\item{ate}{See Usage.}
\item{ate_se}{See Usage.}
\item{ate_pval}{See Usage.}
\item{n}{See Usage.}
\item{n_treated}{See Usage.}
\item{p_treat}{See Usage.}
\item{notes}{Defaults to \code{list()}.}
}
\description{
Part of the otis_causal implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
