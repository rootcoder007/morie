% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{placebo_did}
\alias{placebo_did}
\title{Pre-period placebo DiD}
\usage{
placebo_did(Y, D, event_time, split = NULL)
}
\arguments{
\item{Y}{Balanced panel.}
\item{D}{0/1 vector.}
\item{event_time}{\code{H}.}
\item{split}{Pre-period split point.}
}
\description{
Runs the estimator entirely inside the pre-period. Zero is consistent with parallel trends but does not establish it.
}
\value{
A list mirroring the Python \code{RichResult} payload.
}
\keyword{internal}
