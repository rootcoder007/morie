% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{abc_gp_emulator}
\alias{abc_gp_emulator}
\alias{abcgpemulator}
\title{ABC with a GP surrogate, four methods}
\usage{
abc_gp_emulator(sim, obs, X_grid = NULL, kernel = "sqexp", method = "wilkinson", 
    prior_ppf = NULL, log_prior = NULL, theta0 = NULL, n_sim = 50L, epsilon = 1, 
    summary = NULL, n_waves = 3L, n_design = 32L, threshold = 10, n_sd = 3, 
    accept_kernel = "gaussian", n_iter = 200L, proposal_sd = 0.5, xi = 0.05, 
    delta_s = 10L, n_alpha = 64L, seed = 0L) 

abcgpemulator(sim, obs, X_grid = NULL, kernel = "sqexp", method = "wilkinson", 
    prior_ppf = NULL, log_prior = NULL, theta0 = NULL, n_sim = 50L, epsilon = 1, 
    summary = NULL, n_waves = 3L, n_design = 32L, threshold = 10, n_sd = 3, 
    accept_kernel = "gaussian", n_iter = 200L, proposal_sd = 0.5, xi = 0.05, 
    delta_s = 10L, n_alpha = 64L, seed = 0L) 
}
\description{
Wilkinson (2014) history matching + GP-emulated log-posterior on a grid, or one of the Meeds & Welling (2014) samplers.
}
\arguments{
\item{sim}{Argument \code{sim}; see Usage.}
\item{obs}{Argument \code{obs}; see Usage.}
\item{X_grid}{Argument \code{X_grid}; see Usage.}
\item{kernel}{Argument \code{kernel}; see Usage.}
\item{method}{Argument \code{method}; see Usage.}
\item{prior_ppf}{Argument \code{prior_ppf}; see Usage.}
\item{log_prior}{Argument \code{log_prior}; see Usage.}
\item{theta0}{Argument \code{theta0}; see Usage.}
\item{n_sim}{Argument \code{n_sim}; see Usage.}
\item{epsilon}{Argument \code{epsilon}; see Usage.}
\item{summary}{Argument \code{summary}; see Usage.}
\item{n_waves}{Argument \code{n_waves}; see Usage.}
\item{n_design}{Argument \code{n_design}; see Usage.}
\item{threshold}{Argument \code{threshold}; see Usage.}
\item{n_sd}{Argument \code{n_sd}; see Usage.}
\item{accept_kernel}{Argument \code{accept_kernel}; see Usage.}
\item{n_iter}{Argument \code{n_iter}; see Usage.}
\item{proposal_sd}{Argument \code{proposal_sd}; see Usage.}
\item{xi}{Argument \code{xi}; see Usage.}
\item{delta_s}{Argument \code{delta_s}; see Usage.}
\item{n_alpha}{Argument \code{n_alpha}; see Usage.}
\item{seed}{Argument \code{seed}; see Usage.}
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
