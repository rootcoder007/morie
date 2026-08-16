% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.tmldta_logit_irls}
\alias{.tmldta_logit_irls}
\title{Logistic IRLS that returns a coefficient vector for a 0/1 outcome}
\usage{
.tmldta_logit_irls(Z, a, ridge = 1e-08, max_iter = 50L, tol = 1e-10)
}
\arguments{
\item{Z}{See Usage.}
\item{a}{See Usage.}
\item{ridge}{Defaults to \code{1e-08}.}
\item{max_iter}{Defaults to \code{50L}.}
\item{tol}{Defaults to \code{1e-10}.}
}
\description{
Part of the tmldta_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{b}, as built in the body.
}
\keyword{internal}
