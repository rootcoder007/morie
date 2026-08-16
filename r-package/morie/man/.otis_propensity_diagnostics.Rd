% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.otis_propensity_diagnostics}
\alias{.otis_propensity_diagnostics}
\title{Brier + log-loss + observed/predicted prevalence}
\usage{
.otis_propensity_diagnostics(p, d)
}
\arguments{
\item{p}{See Usage.}
\item{d}{See Usage.}
}
\description{
Part of the otis_causal implementation; see the file header for the source it follows.
}
\value{
A list with \code{brier}, \code{obs_prevalence}, \code{predicted_prevalence}, \code{log_loss}.
}
\keyword{internal}
