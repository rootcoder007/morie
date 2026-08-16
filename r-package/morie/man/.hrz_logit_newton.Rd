% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.hrz_logit_newton}
\alias{.hrz_logit_newton}
\title{.hrz_logit_newton}
\usage{
.hrz_logit_newton(D, X, maxit = 50, tol = 1e-08)
}
\arguments{
\item{D}{Numeric; combined arithmetically in the body.}
\item{X}{A matrix; passed to \code{ncol}.}
\item{maxit}{Defaults to \code{50}.}
\item{tol}{Defaults to \code{1e-08}.}
}
\description{
A step of the helpers_horowitz implementation. Called by \code{hrzt1}. See the file header for the source the module follows.
}
\value{
A numeric value.
}
\keyword{internal}
