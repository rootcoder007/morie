% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{denoise}
\alias{denoise}
\title{Reverse chain}
\usage{
denoise(x_t, model, schedule, t_start = NULL)
}
\arguments{
\item{x_t}{Noisy state to start from.}
\item{model}{Function \code{(x, t)} returning the model\'s   \code{x_0} estimate.}
\item{schedule}{Schedule list from \code{noise_schedule}.}
\item{t_start}{Starting timestep; defaults to the last step.}
}
\description{
Walks the schedule from \code{t} back to 0. With \code{scale = 0} the schedule is the identity and this returns its input unchanged.
}
\value{
A list mirroring the Python \code{RichResult} payload.
}
\keyword{internal}
