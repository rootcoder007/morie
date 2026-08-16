% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_tps_psdp_feature_query}
\alias{.morie_tps_psdp_feature_query}
\title{Internal: GET a TPS PSDP ArcGIS FeatureServer layer; return a}
\usage{
.morie_tps_psdp_feature_query(layer_url, where = "1=1", max_features = NULL, return_geometry = FALSE)
}
\arguments{
\item{layer_url}{Passed to \code{.morie_dataset_tps_fetch}.}
\item{where}{Passed to \code{.morie_dataset_tps_fetch}. Defaults to \code{"1=1"}.}
\item{max_features}{Passed to \code{.morie_dataset_tps_fetch}.}
\item{return_geometry}{Passed to \code{.morie_dataset_tps_fetch}. Defaults to \code{FALSE}.}
}
\description{
data.frame of attributes. Mock this in tests.
}
\value{
Nothing; this branch always raises.
}
\keyword{internal}
