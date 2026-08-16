% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.ms_baseline}
\alias{.ms_baseline}
\title{Breslow cumulative baseline hazard, what survival::basehaz returns}
\usage{
.ms_baseline(time, event, X, beta)
}
\arguments{
\item{time}{Passed to \code{.ms_check}.}
\item{event}{Passed to \code{.ms_check}.}
\item{X}{A matrix; passed to \code{nrow}.}
\item{beta}{A matrix; passed to \code{\%*\%}.}
}
\description{
Part of the survival_more implementation; see the file header for the source it follows.
}
\value{
A list with \code{time}, \code{cumhaz}, \code{weight}.
}
\keyword{internal}
