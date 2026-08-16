% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{history_match}
\alias{history_match}
\title{Sequential history matching}
\usage{
history_match(sim, obs, prior_ppf, n_waves = 3L, n_design = 32L, n_sim = 50L, epsilon = 1, summary = NULL, threshold = 10, n_sd = 3, kernel = "sqexp", accept_kernel = "gaussian", seed = 0L)
}
\description{
Wilkinson (2014) Sec. 3: waves of design, emulate, rule out, redesign. Returns the GP fit and a list of per-wave summaries.
}
\value{
A list with \code{fit}, \code{waves}.
}
\keyword{internal}
