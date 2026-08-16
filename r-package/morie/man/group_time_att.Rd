% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{group_time_att}
\alias{group_time_att}
\title{Group-time ATT(g,t) under staggered adoption}
\usage{
group_time_att(Y, first_treated, comparison = "not-yet-treated")
}
\arguments{
\item{Y}{Balanced panel.}
\item{first_treated}{Numeric vector of adoption periods (1-based),   with \code{NA} or \code{Inf} for never-treated.}
\item{comparison}{\code{"never-treated"} or   \code{"not-yet-treated"}.}
}
\description{
Group-time ATT(g,t) under staggered adoption
}
\value{
A list with \code{att} (named list keyed by \code{(g,t)}),   \code{cohorts}, \code{T}, \code{n}, \code{comparison},   \code{estimate}, \code{method}.
}
\keyword{internal}
