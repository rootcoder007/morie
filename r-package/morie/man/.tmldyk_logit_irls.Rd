% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.tmldyk_logit_irls}
\alias{.tmldyk_logit_irls}
\title{Logistic IRLS that returns a coefficient vector}
\usage{
.tmldyk_logit_irls(Z, a, ridge = 1e-08, max_iter = 50L, tol = 1e-10)
}
\arguments{
\item{Z}{See Usage.}
\item{a}{A vector; its length is taken.}
\item{ridge}{Numeric; combined arithmetically in the body. Defaults to \code{1e-08}.}
\item{max_iter}{A count; the body uses it as \code{seq_len(...)}. Defaults to \code{50L}.}
\item{tol}{Defaults to \code{1e-10}.}
}
\description{
Part of the tmldyk_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{b}, as built in the body.
}
\keyword{internal}
