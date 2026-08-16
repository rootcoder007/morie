% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_read_text}
\alias{.morie_read_text}
\title{Read text from a URL (used for JSON/XML/HTML API responses)}
\usage{
.morie_read_text(url)
}
\arguments{
\item{url}{Passed to \code{url}.}
}
\description{
A step of the data_access implementation. Called by \code{morie_ckan_search}, \code{morie_fetch}, \code{morie_fetch_arcgis}. See the file header for the source the module follows.
}
\value{
A character value.
}
\keyword{internal}
