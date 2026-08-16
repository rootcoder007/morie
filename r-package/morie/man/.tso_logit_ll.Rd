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
\item{eta}{See Usage.}
\item{y}{See Usage.}
}
\description{
(intercept already folded in).  Uses log1p(exp(-|eta|)) for stability.
}
\value{
A numeric value.
}
\keyword{internal}
