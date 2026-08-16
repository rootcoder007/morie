% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.tps_stoch_neg_loglik_hawkes}
\alias{.tps_stoch_neg_loglik_hawkes}
\title{Negative log-likelihood of exponential-kernel Hawkes:}
\usage{
.tps_stoch_neg_loglik_hawkes(params, t, T_window)
}
\arguments{
\item{params}{A vector; indexed elementwise.}
\item{t}{A vector; its length is taken and its elements indexed.}
\item{T_window}{Numeric; combined arithmetically in the body.}
}
\description{
lambda(t) = mu + kappa*omega * sum_{t_i<t} exp(-omega*(t - t_i)) Closed-form integral: mu*T + kappa * sum_i (1 - exp(-omega*(T - t_i)))
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.tps_stoch_result(V, V)
.tps_stoch_neg_loglik_hawkes(params = V, t = V, T_window = V)
}
