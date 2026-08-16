% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_url_with_params}
\alias{.morie_url_with_params}
\title{Append a named list of query parameters to a URL, URL-encoding values}
\usage{
.morie_url_with_params(url, params = NULL)
}
\arguments{
\item{url}{Character; passed to \code{grepl}.}
\item{params}{Optional; may be \code{NULL}. A vector; its length is taken and its elements indexed.}
}
\description{
A step of the data_access implementation. Called by \code{morie_ckan_search}, \code{morie_fetch}, \code{morie_fetch_arcgis}. See the file header for the source the module follows.
}
\value{
A character value.
}
\keyword{internal}
