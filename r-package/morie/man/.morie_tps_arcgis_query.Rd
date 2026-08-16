% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_tps_arcgis_query}
\alias{.morie_tps_arcgis_query}
\title{Internal: a single ArcGIS FeatureServer /query call}
\usage{
.morie_tps_arcgis_query(layer_url, where = "1=1", out_fields = "*", return_geometry = FALSE, result_offset = 0L, result_record_count = 2000L, user_agent = .MORIE_TPS_DEFAULT_UA, timeout = .MORIE_TPS_DEFAULT_TIMEOUT)
}
\arguments{
\item{layer_url}{See Usage.}
\item{where}{Defaults to \code{"1=1"}.}
\item{out_fields}{Defaults to \code{"*"}.}
\item{return_geometry}{A flag; the body branches on it. Defaults to \code{FALSE}.}
\item{result_offset}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{0L}.}
\item{result_record_count}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{2000L}.}
\item{user_agent}{Defaults to \code{.MORIE_TPS_DEFAULT_UA}.}
\item{timeout}{Coerced to integer by the body, with \code{as.integer}.}
}
\description{
A step of the ingest_tps implementation. Called by \code{morie_ingest_tps_feature_layer}. See the file header for the source the module follows.
}
\value{
The value of \code{payload}, as built in the body.
}
\keyword{internal}
