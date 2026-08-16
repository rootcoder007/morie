% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.hawkes_loglik_poisson}
\alias{.hawkes_loglik_poisson}
\title{Closed-form log-likelihood of the homogeneous Poisson submodel}
\usage{
.hawkes_loglik_poisson(n, end_time)
}
\arguments{
\item{n}{See Usage.}
\item{end_time}{See Usage.}
}
\description{
(eta = 0): its MLE baseline is nu_hat = n / end_time, giving logLik = n*log(n/T) - n. The Hawkes family nests this, so the Hawkes MLE log-likelihood can never fall below it.
}
\value{
A numeric value.
}
\keyword{internal}
