% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_ckan_call}
\alias{.morie_ckan_call}
\title{Internal: perform a CKAN Action-API call and unwrap `result`}
\usage{
.morie_ckan_call(portal, action, params = NULL, api_key = NULL, user_agent = .MORIE_CKAN_DEFAULT_UA, timeout = .MORIE_CKAN_DEFAULT_TIMEOUT)
}
\arguments{
\item{portal}{See Usage.}
\item{action}{See Usage.}
\item{params}{Defaults to \code{NULL}.}
\item{api_key}{Defaults to \code{NULL}.}
\item{user_agent}{Defaults to \code{.MORIE_CKAN_DEFAULT_UA}.}
\item{timeout}{Defaults to \code{.MORIE_CKAN_DEFAULT_TIMEOUT}.}
}
\description{
3YY: collapsed .morie_ckan_build_req + _call into a single function that routes through .morie_dataset_http_text (libcurl with httr2 fallback) + .morie_from_json(simplifyVector=FALSE).
}
\value{
The value of \code{$}.
}
\keyword{internal}
