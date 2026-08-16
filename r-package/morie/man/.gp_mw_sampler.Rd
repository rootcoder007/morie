% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.gp_mw_sampler}
\alias{.gp_mw_sampler}
\title{.gp_mw_sampler}
\usage{
.gp_mw_sampler(sim, obs, log_prior, theta0, n_iter, n_sim, epsilon, proposal_sd, summary, seed, adaptive, xi, delta_s, n_alpha, max_sim = NULL)
}
\arguments{
\item{sim}{See Usage.}
\item{obs}{Passed to \code{.gp_summarise}.}
\item{log_prior}{Passed to \code{.gp_alpha_terms}.}
\item{theta0}{See Usage.}
\item{n_iter}{See Usage.}
\item{n_sim}{See Usage.}
\item{epsilon}{See Usage.}
\item{proposal_sd}{See Usage.}
\item{summary}{Passed to \code{.gp_summarise}.}
\item{seed}{Passed to \code{.ghc_rng}.}
\item{adaptive}{A flag; the body branches on it.}
\item{xi}{See Usage.}
\item{delta_s}{See Usage.}
\item{n_alpha}{A count; the body uses it as \code{seq_len(...)}.}
\item{max_sim}{Defaults to \code{NULL}.}
}
\description{
A step of the abcgp_native implementation. Called by \code{gps_abc}, \code{synthetic_abc}. See the file header for the source the module follows.
}
\value{
A list with \code{chain}, \code{acceptance_rate}, \code{n_simulations}, \code{unresolved_steps}.
}
\keyword{internal}
