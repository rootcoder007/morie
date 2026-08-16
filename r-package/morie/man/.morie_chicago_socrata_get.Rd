% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_chicago_socrata_get}
\alias{.morie_chicago_socrata_get}
\title{Internal: a single Socrata SoQL GET against `resource_url`}
\usage{
.morie_chicago_socrata_get(resource_url, where = NULL, select = NULL, order = NULL, limit = .MORIE_CHICAGO_DEFAULT_PAGE_SIZE, offset = 0L, app_token = NULL, user_agent = .MORIE_CHICAGO_DEFAULT_UA, timeout = .MORIE_CHICAGO_DEFAULT_TIMEOUT)
}
\arguments{
\item{resource_url}{See Usage.}
\item{where}{Defaults to \code{NULL}.}
\item{select}{Defaults to \code{NULL}.}
\item{order}{Defaults to \code{NULL}.}
\item{limit}{Defaults to \code{.MORIE_CHICAGO_DEFAULT_PAGE_SIZE}.}
\item{offset}{Defaults to \code{0L}.}
\item{app_token}{Defaults to \code{NULL}.}
\item{user_agent}{Defaults to \code{.MORIE_CHICAGO_DEFAULT_UA}.}
\item{timeout}{Defaults to \code{.MORIE_CHICAGO_DEFAULT_TIMEOUT}.}
}
\description{
Part of the ingest_chicago implementation; see the file header for the source it follows.
}
\value{
The value of \code{payload}, as built in the body.
}
\keyword{internal}
