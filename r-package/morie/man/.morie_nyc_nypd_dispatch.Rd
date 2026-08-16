% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_nyc_nypd_dispatch}
\alias{.morie_nyc_nypd_dispatch}
\title{.morie_nyc_nypd_dispatch}
\usage{
.morie_nyc_nypd_dispatch(dataset_key, year, max_features, offline, resource_id, mode = c("soda2", "soda3"), paginate = FALSE, page_size = 1000L, max_pages = 200L, app_token = NULL)
}
\arguments{
\item{dataset_key}{See Usage.}
\item{year}{Optional; may be \code{NULL}. Coerced to integer by the body, with \code{as.integer}.}
\item{max_features}{Optional; may be \code{NULL}. Passed to \code{.morie_dataset_socrata_fetch}.}
\item{offline}{A flag; the body branches on it.}
\item{resource_id}{Optional; may be \code{NULL}. Passed to \code{.morie_dataset_soda3_query}.}
\item{mode}{Compared against \code{"soda2"}. Defaults to \code{c("soda2", "soda3")}.}
\item{paginate}{Passed to \code{.morie_dataset_socrata_fetch}. Defaults to \code{FALSE}.}
\item{page_size}{Passed to \code{.morie_dataset_socrata_fetch}. Defaults to \code{1000L}.}
\item{max_pages}{Passed to \code{.morie_dataset_socrata_fetch}. Defaults to \code{200L}.}
\item{app_token}{Passed to \code{.morie_dataset_soda3_query}.}
}
\description{
A step of the datasets_nyc_nypd implementation. Called by \code{morie_datasets_nyc_nypd_arrests_historic}, \code{morie_datasets_nyc_nypd_arrests_ytd}, \code{morie_datasets_nyc_nypd_by_key} and 6 others in the module. See the file header for the source the module follows.
}
\value{
The value of \code{.morie_dataset_soda3_query}.
}
\keyword{internal}
