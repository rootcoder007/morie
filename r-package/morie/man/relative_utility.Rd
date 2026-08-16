% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{relative_utility}
\alias{relative_utility}
\title{Utility of one standard quarantine relative to another}
\usage{
relative_utility(t_R_a, t_R_b, t_Q = 3, generation_time = NULL, infected_fraction = NULL)
}
\arguments{
\item{t_R_a}{Release time of strategy A.}
\item{t_R_b}{Release time of strategy B.}
\item{t_Q}{Quarantine start.}
\item{generation_time}{Output of \code{\link{gamma_generation_time}}.}
\item{infected_fraction}{Optional, ignored in the computation.}
}
\description{
The infected fraction cancels in the ratio for standard quarantine; \code{infected_fraction} is accepted only to demonstrate the cancellation.
}
\value{
A list with \code{relative_utility}, \code{utility_a},   \code{utility_b}, \code{efficacy_a}, \code{efficacy_b},   \code{independent_of_infected_fraction}, \code{note}.
}
\keyword{internal}
