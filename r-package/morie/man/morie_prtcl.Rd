% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_prtcl}
\alias{morie_prtcl}
\alias{particlefilter}
\title{Bootstrap particle filter (Algorithm 1 of King, Nguyen & Ionides 2016)}
\usage{
morie_prtcl(y, n_particles, init, step, loglik, seed = 0L, resample_threshold = 1, 
    systematic = TRUE) 

particlefilter(y, n_particles, init, step, loglik, seed = 0L, resample_threshold = 1, 
    systematic = TRUE) 
}
\arguments{
\item{y}{Numeric vector of observations.}
\item{n_particles}{Integer, number of particles.}
\item{init}{Function \code{(rng) -> state} for one particle.}
\item{step}{Function \code{(state, t, rng) -> state} for the process.}
\item{loglik}{Function \code{(state, obs, t) -> float} for the   measurement density.}
\item{seed}{Seed for the shared generator.}
\item{resample_threshold}{Resample when ESS drops below this   fraction of \code{n_particles}; default 1.0 (every step).}
\item{systematic}{If \code{TRUE} use systematic resampling   (Algorithm 2); otherwise multinomial.}
}
\description{
Propagate each particle, weight by the measurement density, resample when the ESS falls below \code{resample_threshold * J}.
}
\value{
A list with \code{estimate}, \code{filtered_mean},   \code{loglik}, \code{ess}, \code{min_ess}, \code{resampled},   \code{n_particles}, \code{n_obs}, \code{systematic},   \code{particles}, \code{method}.
}
\keyword{internal}
