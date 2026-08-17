% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{abc_gp_emulator}
\alias{abc_gp_emulator}
\title{ABC with a GP surrogate, four methods}
\usage{
abc_gp_emulator(sim, obs, X_grid = NULL, kernel = "sqexp", method = "wilkinson", prior_ppf = NULL, log_prior = NULL, theta0 = NULL, n_sim = 50L, epsilon = 1, summary = NULL, n_waves = 3L, n_design = 32L, threshold = 10, n_sd = 3, accept_kernel = "gaussian", n_iter = 200L, proposal_sd = 0.5, xi = 0.05, delta_s = 10L, n_alpha = 64L, seed = 0L)
}
\description{
Wilkinson (2014) history matching + GP-emulated log-posterior on a grid, or one of the Meeds & Welling (2014) samplers.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
\examples{
sim <- function(theta, e) rnorm(1, mean = theta[1])
em <- abc_gp_emulator(sim, obs = 0.3, method = "wilkinson",
                      prior_ppf = list(-2, 2), n_waves = 2L,
                      n_design = 8L, n_sim = 20L, seed = 1L)
is.list(em)
}
