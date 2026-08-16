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
\item{resource_url}{Passed to \code{.morie_dataset_http_text}.}
\item{where}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
\item{select}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
\item{order}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
\item{limit}{Coerced to integer by the body, with \code{as.integer}.}
\item{offset}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{0L}.}
\item{app_token}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
\item{user_agent}{Accepted by the signature and not used anywhere in the body.}
\item{timeout}{Coerced to integer by the body, with \code{as.integer}.}
}
\description{
A step of the ingest_chicago implementation. Called by \code{morie_ingest_chicago_socrata}. See the file header for the source the module follows.
}
\value{
The value of \code{payload}, as built in the body.
}
\keyword{internal}
