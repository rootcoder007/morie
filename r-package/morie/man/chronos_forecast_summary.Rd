% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{chronos_forecast_summary}
\alias{chronos_forecast_summary}
\title{chronos_forecast_summary}
\usage{
chronos_forecast_summary(token_probs, bins, quantiles = c(0.1, 0.5, 0.9))
}
\arguments{
\item{token_probs}{See Usage.}
\item{bins}{See Usage.}
\item{quantiles}{Defaults to \code{c(0.1, 0.5, 0.9)}.}
}
\description{
Part of the chronos_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{mean}, \code{quantiles}, \code{mode}, \code{note}.
}
\keyword{internal}
