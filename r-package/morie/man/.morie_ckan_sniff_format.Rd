% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_ckan_sniff_format}
\alias{.morie_ckan_sniff_format}
\title{Internal: sniff a resource format from URL extension when unset}
\usage{
.morie_ckan_sniff_format(url, as_format = NULL)
}
\arguments{
\item{url}{Character; passed to \code{sub}.}
\item{as_format}{Optional; may be \code{NULL}. Character; passed to \code{tolower}.}
}
\description{
A step of the ingest_ckan implementation. Called by \code{morie_ingest_ckan_read_resource}. See the file header for the source the module follows.
}
\value{
The value of \code{tolower}.
}
\keyword{internal}
