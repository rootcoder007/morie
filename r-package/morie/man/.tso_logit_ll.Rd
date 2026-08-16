% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.tso_logit_ll}
\alias{.tso_logit_ll}
\title{Log-likelihood of a single binary logit with linear predictor `eta`}
\usage{
.tso_logit_ll(eta, y)
}
\arguments{
\item{eta}{Numeric; passed to \code{exp}.}
\item{y}{Numeric; combined arithmetically in the body.}
}
\description{
(intercept already folded in).  Uses log1p(exp(-|eta|)) for stability.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.tso_logit_ll(V, V)
}
