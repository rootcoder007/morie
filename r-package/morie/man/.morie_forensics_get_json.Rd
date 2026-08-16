% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_forensics_get_json}
\alias{.morie_forensics_get_json}
\title{Internal: shared GET + JSON parse with explicit auth checks}
\usage{
.morie_forensics_get_json(url, params = list(), headers = list(), timeout = .MORIE_FORENSICS_DEFAULT_TIMEOUT, user_agent = .MORIE_FORENSICS_DEFAULT_UA, auth_signup_url = NULL, label = "forensics")
}
\arguments{
\item{url}{Passed to \code{.morie_dataset_http_text_with_status}.}
\item{params}{Passed to \code{.morie_dataset_http_text_with_status}. Defaults to \code{list()}.}
\item{headers}{A vector; its length is taken. Defaults to \code{list()}.}
\item{timeout}{Coerced to integer by the body, with \code{as.integer}.}
\item{user_agent}{Accepted by the signature and not used anywhere in the body.}
\item{auth_signup_url}{Defaults to \code{NULL}.}
\item{label}{Defaults to \code{"forensics"}.}
}
\description{
3ZZ: routes through .morie_dataset_http_text_with_status (libcurl backend with httr2 fallback), inspects HTTP status code for 401/403 (auth) + 4xx (generic) custom error formatting.
}
\value{
The value of \code{.morie_from_json}.
}
\keyword{internal}
