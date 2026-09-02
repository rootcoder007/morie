% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{history_match}
\alias{history_match}
\title{Sequential history matching}
\usage{
history_match(sim, obs, prior_ppf, n_waves = 3L, n_design = 32L, n_sim = 50L, 
    epsilon = 1, summary = NULL, threshold = 10, n_sd = 3, kernel = "sqexp", 
    accept_kernel = "gaussian", seed = 0L) 
}
\description{
Wilkinson (2014) Sec. 3: waves of design, emulate, rule out, redesign. Returns the GP fit and a list of per-wave summaries.
}
\arguments{
\item{sim}{Argument \code{sim}; see Usage.}
\item{obs}{Argument \code{obs}; see Usage.}
\item{prior_ppf}{Argument \code{prior_ppf}; see Usage.}
\item{n_waves}{Argument \code{n_waves}; see Usage.}
\item{n_design}{Argument \code{n_design}; see Usage.}
\item{n_sim}{Argument \code{n_sim}; see Usage.}
\item{epsilon}{Argument \code{epsilon}; see Usage.}
\item{summary}{Argument \code{summary}; see Usage.}
\item{threshold}{Argument \code{threshold}; see Usage.}
\item{n_sd}{Argument \code{n_sd}; see Usage.}
\item{kernel}{Argument \code{kernel}; see Usage.}
\item{accept_kernel}{Argument \code{accept_kernel}; see Usage.}
\item{seed}{Argument \code{seed}; see Usage.}
}
\value{
A list with \code{fit}, \code{waves}.
}
\keyword{internal}
\examples{
sim <- function(theta, e) rnorm(1, mean = theta[1])
hm <- history_match(sim, obs = 0.3, prior_ppf = list(-2, 2),
                    n_waves = 2L, n_design = 8L, n_sim = 20L, seed = 1L)
length(hm$waves) == 2L
}
