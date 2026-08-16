% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_otis_lookup_pending_dispatch}
\alias{.morie_otis_lookup_pending_dispatch}
\title{Internal shared dispatch for the OTIS wrappers. Resource ids are}
\usage{
.morie_otis_lookup_pending_dispatch(dataset_label, fixture, offline, resource_id, registry_key = NULL, source = NULL)
}
\arguments{
\item{dataset_label}{See Usage.}
\item{fixture}{See Usage.}
\item{offline}{See Usage.}
\item{resource_id}{See Usage.}
\item{registry_key}{Defaults to \code{NULL}.}
\item{source}{Defaults to \code{NULL}.}
}
\description{
auto-resolved from .MORIE_ONTARIO_CKAN_REGISTRY[[registry_key]] when the caller doesn\\\'t pass an explicit override; if the registry entry is also missing or NA the function errors with a clear message.
}
\value{
The value of \code{.morie_load_chain}.
}
\keyword{internal}
