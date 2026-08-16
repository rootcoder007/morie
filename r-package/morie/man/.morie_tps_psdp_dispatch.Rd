% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_tps_psdp_dispatch}
\alias{.morie_tps_psdp_dispatch}
\title{.morie_tps_psdp_dispatch}
\usage{
.morie_tps_psdp_dispatch(layer_key, year, max_features, offline, layer_url)
}
\arguments{
\item{layer_key}{See Usage.}
\item{year}{See Usage.}
\item{max_features}{Optional; may be \code{NULL}. Passed to \code{.morie_tps_psdp_feature_query}.}
\item{offline}{A flag; the body branches on it.}
\item{layer_url}{Optional; may be \code{NULL}. Passed to \code{.morie_tps_psdp_feature_query}.}
}
\description{
A step of the datasets_tps_psdp implementation. Called by \code{morie_datasets_tps_assault}, \code{morie_datasets_tps_autotheft}, \code{morie_datasets_tps_bicycletheft} and 9 others in the module. See the file header for the source the module follows.
}
\value{
The value of \code{morie_datasets_tps_arcgis_hub_by_id}.
}
\keyword{internal}
