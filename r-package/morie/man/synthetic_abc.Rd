% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{synthetic_abc}
\alias{synthetic_abc}
\title{Synthetic-likelihood ABC, Meeds & Welling Algorithm 1}
\usage{
synthetic_abc(sim, obs, log_prior, theta0, n_iter = 200L, n_sim = 20L, epsilon = 0, 
    proposal_sd = 0.5, summary = NULL, seed = 0L) 
}
\description{
Synthetic-likelihood ABC, Meeds & Welling Algorithm 1
}
\arguments{
\item{sim}{Argument \code{sim}; see Usage.}
\item{obs}{Argument \code{obs}; see Usage.}
\item{log_prior}{Argument \code{log_prior}; see Usage.}
\item{theta0}{Argument \code{theta0}; see Usage.}
\item{n_iter}{Argument \code{n_iter}; see Usage.}
\item{n_sim}{Argument \code{n_sim}; see Usage.}
\item{epsilon}{Argument \code{epsilon}; see Usage.}
\item{proposal_sd}{Argument \code{proposal_sd}; see Usage.}
\item{summary}{Argument \code{summary}; see Usage.}
\item{seed}{Argument \code{seed}; see Usage.}
}
\value{
The value of \code{.gp_mw_sampler}.
}
\keyword{internal}
\examples{
sim <- function(theta, e) rnorm(1, mean = theta[1])
log_prior <- function(theta) sum(dnorm(theta, 0, 2, log = TRUE))
r <- synthetic_abc(sim, obs = 0.3, log_prior, theta0 = 0,
                   n_iter = 30L, n_sim = 10L, epsilon = 1, seed = 1L)
nrow(as.matrix(r$chain)) >= 1L
}
