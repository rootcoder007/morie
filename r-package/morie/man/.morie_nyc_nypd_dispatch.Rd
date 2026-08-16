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
\item{year}{See Usage.}
\item{max_features}{See Usage.}
\item{offline}{See Usage.}
\item{resource_id}{See Usage.}
\item{mode}{Defaults to \code{c("soda2", "soda3")}.}
\item{paginate}{Defaults to \code{FALSE}.}
\item{page_size}{Defaults to \code{1000L}.}
\item{max_pages}{Defaults to \code{200L}.}
\item{app_token}{Defaults to \code{NULL}.}
}
\description{
Part of the datasets_nyc_nypd implementation; see the file header for the source it follows.
}
\value{
The value of \code{.morie_dataset_soda3_query}.
}
\keyword{internal}
