% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{aggregate_att}
\alias{aggregate_att}
\title{Aggregate ATT(g,t)}
\usage{
aggregate_att(gt, scheme = "simple", horizon = NULL)
}
\arguments{
\item{gt}{Result of \code{group_time_att}.}
\item{scheme}{\code{"simple"}, \code{"event"} or   \code{"cohort"}.}
\item{horizon}{Optional event-time cap for \code{"event"}.}
}
\description{
Aggregate ATT(g,t)
}
\value{
A list with \code{estimate}, \code{scheme} (and   \code{profile} for the by-event-time and by-cohort schemes).
}
\keyword{internal}
