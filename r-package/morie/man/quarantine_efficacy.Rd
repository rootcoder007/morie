% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{quarantine_efficacy}
\alias{quarantine_efficacy}
\title{Quarantine efficacy (eq. 1 of Ashcroft et al. 2021)}
\usage{
quarantine_efficacy(t_Q, t_R, generation_time = NULL, t_E = 0)
}
\arguments{
\item{t_Q}{Quarantine start.}
\item{t_R}{Release time.}
\item{generation_time}{Output of \code{\link{gamma_generation_time}}.}
\item{t_E}{Exposure time.}
}
\description{
\code{efficacy = mass of the density in [t_Q, t_R] / mass remaining after t_Q}. Transmission before \code{t_Q} is already gone, so the denominator is the mass from \code{t_Q} on.
}
\value{
A list with \code{efficacy}, \code{prevented_mass},   \code{remaining_mass}, \code{t_Q}, \code{t_R}, \code{max_attainable},   \code{pre_quarantine_mass} (and a \code{note} if no transmission   remains).
}
\keyword{internal}
\examples{
quarantine_efficacy(t_Q = 5L, t_R = 5L)
}
