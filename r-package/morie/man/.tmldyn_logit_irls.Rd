% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.tmldyn_logit_irls}
\alias{.tmldyn_logit_irls}
\title{Logistic IRLS for binary outcome, used by the intervention mechanism}
\usage{
.tmldyn_logit_irls(X, a, ridge = 1e-08, max_iter = 50L, tol = 1e-10, penalty = 0)
}
\arguments{
\item{X}{A matrix; the body checks with \code{is.matrix}.}
\item{a}{Numeric; combined arithmetically in the body.}
\item{ridge}{Numeric; combined arithmetically in the body. Defaults to \code{1e-08}.}
\item{max_iter}{A count; the body uses it as \code{seq_len(...)}. Defaults to \code{50L}.}
\item{tol}{Defaults to \code{1e-10}.}
}
\description{
A step of the tmldyn_native implementation. Called by \code{.intervention_mechanism}. See the file header for the source the module follows. source it follows.
}
\value{
The value of \code{b}, as built in the body.
}
\keyword{internal}
