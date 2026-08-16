% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{noise_schedule}
\alias{noise_schedule}
\title{Linear noise schedule, scaled down}
\usage{
noise_schedule(T, scale = 0.001, beta_min = 1e-04, beta_max = 0.02)
}
\arguments{
\item{T}{Number of diffusion steps.}
\item{scale}{Noise scale multiplier.}
\item{beta_min}{Lower endpoint of the unscaled linear schedule.}
\item{beta_max}{Upper endpoint of the unscaled linear schedule.}
}
\description{
At image-diffusion scales the personal history is destroyed. The \code{scale} knob keeps it recoverable.
}
\value{
A list with \code{beta}, \code{alpha_bar}, \code{T},   \code{scale}, \code{signal_retained}, \code{note}.
}
\keyword{internal}
\examples{
noise_schedule(T = 5L)
}
