% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_tps_fetch_arcgis_query}
\alias{.morie_tps_fetch_arcgis_query}
\title{Internal: one ArcGIS REST /query GET. Returns the parsed GeoJSON}
\usage{
.morie_tps_fetch_arcgis_query(base_url, where, offset, max_records = 2000L, timeout = 120)
}
\arguments{
\item{base_url}{See Usage.}
\item{where}{See Usage.}
\item{offset}{See Usage.}
\item{max_records}{Defaults to \code{2000L}.}
\item{timeout}{Defaults to \code{120}.}
}
\description{
list (raises on HTTP / JSON failure).
}
\value{
The value of \code{.morie_from_json}.
}
\keyword{internal}
