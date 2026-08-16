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
\item{portal}{Passed to \code{.morie_ckan_portal}.}
\item{action}{See Usage.}
\item{params}{Passed to \code{.morie_dataset_http_text}.}
\item{api_key}{Defaults to \code{NULL}.}
\item{user_agent}{Accepted by the signature and not used anywhere in the body. Defaults to \code{.MORIE_CKAN_DEFAULT_UA}.}
\item{timeout}{Coerced to integer by the body, with \code{as.integer}.}
}
\description{
3YY: collapsed .morie_ckan_build_req + _call into a single function that routes through .morie_dataset_http_text (libcurl with httr2 fallback) + .morie_from_json(simplifyVector=FALSE).
}
\value{
The value of \code{$}.
}
\keyword{internal}
