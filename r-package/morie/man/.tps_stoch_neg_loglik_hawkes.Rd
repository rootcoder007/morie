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
\item{params}{See Usage.}
\item{t}{See Usage.}
\item{T_window}{See Usage.}
}
\description{
lambda(t) = mu + kappa*omega * sum_{t_i<t} exp(-omega*(t - t_i)) Closed-form integral: mu*T + kappa * sum_i (1 - exp(-omega*(T - t_i)))
}
\value{
A numeric value.
}
\keyword{internal}
