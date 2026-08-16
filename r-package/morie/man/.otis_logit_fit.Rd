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
\item{X}{A matrix; passed to \code{nrow}.}
\item{d}{Numeric; combined arithmetically in the body.}
\item{ridge}{Numeric; combined arithmetically in the body. Defaults to \code{0.001}.}
\item{max_iter}{A count; the body uses it as \code{seq_len(...)}. Defaults to \code{50L}.}
\item{tol}{Passed to \code{<}. Defaults to \code{1e-06}.}
}
\description{
Part of the otis_causal implementation; see the file header for the source it follows.
}
\value{
The value of \code{beta}, as built in the body.
}
\keyword{internal}
\examples{
.otis_logit_fit(X = matrix(c(1, 2, 3, 4, 5, 6), nrow = 2), d = 5L)
}
