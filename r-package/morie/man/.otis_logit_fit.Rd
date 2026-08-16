% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.otis_logit_fit}
\alias{.otis_logit_fit}
\title{Newton-Raphson logistic with ridge penalty}
\usage{
.otis_logit_fit(X, d, ridge = 0.001, max_iter = 50L, tol = 1e-06)
}
\arguments{
\item{X}{See Usage.}
\item{d}{See Usage.}
\item{ridge}{Defaults to \code{0.001}.}
\item{max_iter}{Defaults to \code{50L}.}
\item{tol}{Defaults to \code{1e-06}.}
}
\description{
Part of the otis_causal implementation; see the file header for the source it follows.
}
\value{
The value of \code{beta}, as built in the body.
}
\keyword{internal}
