% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.aalen_johansen}
\alias{.aalen_johansen}
\title{Aalen-Johansen estimator of F_k(t) plus the overall KM survival}
\usage{
.aalen_johansen(time, event_type, cause = 1)
}
\arguments{
\item{time}{Passed to \code{.s03vec}.}
\item{event_type}{Passed to \code{.s03vec}.}
\item{cause}{Defaults to \code{1}.}
}
\description{
F_k(t) = sum_{u <= t} S(u-) dN_k(u) / Y(u).  The S(u-) factor is the ALL-CAUSE Kaplan-Meier: leaving it out gives 1 - KM computed on the cause alone, which over-states the incidence whenever a competing event can happen first.
}
\value{
A list with \code{times}, \code{F}, \code{S}, \code{Y}, \code{dk}, \code{n}.
}
\keyword{internal}
