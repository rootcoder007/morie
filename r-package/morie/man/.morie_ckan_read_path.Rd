% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_ckan_read_path}
\alias{.morie_ckan_read_path}
\title{Internal: read a downloaded resource path into a data.frame by format}
\usage{
.morie_ckan_read_path(path, fmt)
}
\arguments{
\item{path}{Passed to \code{.morie_from_json}.}
\item{fmt}{One of \code{"csv"}, \code{"json"}, \code{"parquet"}, \code{"tab"}, \code{"tsv"}, \code{"xls"}, \code{"xlsx"}.}
}
\description{
A step of the ingest_ckan implementation. Called by \code{morie_ingest_ckan_read_resource}. See the file header for the source the module follows.
}
\value{
The value of \code{utils::read.csv}.
}
\keyword{internal}
