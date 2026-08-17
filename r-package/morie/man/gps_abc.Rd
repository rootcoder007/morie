% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{gps_abc}
\alias{gps_abc}
\title{GPS-ABC adaptive sampler, Meeds & Welling Algorithm 2}
\usage{
gps_abc(sim, obs, log_prior, theta0, n_iter = 200L, n_sim = 10L, epsilon = 0, proposal_sd = 0.5, summary = NULL, seed = 0L, xi = 0.05, delta_s = 10L, n_alpha = 64L, max_sim = 400L)
}
\description{
GPS-ABC adaptive sampler, Meeds & Welling Algorithm 2
}
\value{
The value of \code{.gp_mw_sampler}.
}
\keyword{internal}
\examples{
sim <- function(theta, e) rnorm(1, mean = theta[1])
log_prior <- function(theta) sum(dnorm(theta, 0, 2, log = TRUE))
r <- gps_abc(sim, obs = 0.3, log_prior, theta0 = 0, n_iter = 30L,
             n_sim = 8L, epsilon = 1, seed = 1L)
is.list(r)
}
