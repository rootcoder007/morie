% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_arsau_uof_dispatch}
\alias{.morie_arsau_uof_dispatch}
\title{Internal: shared offline+live dispatch for ARSAU UoF wrappers}
\usage{
.morie_arsau_uof_dispatch(kind, year, offline, resource_id, fixture_name)
}
\arguments{
\item{kind}{See Usage.}
\item{year}{Coerced to character by the body, with \code{as.character}.}
\item{offline}{A flag; the body branches on it.}
\item{resource_id}{Optional; may be \code{NULL}. Passed to \code{.morie_ontario_ckan_dump_csv}.}
\item{fixture_name}{See Usage.}
}
\description{
A step of the datasets_ontario_tps implementation. Called by \code{morie_datasets_arsau_aggregate_summary}, \code{morie_datasets_arsau_detailed_dataset}, \code{morie_datasets_arsau_uof_individual_records} and 2 others in the module. See the file header for the source the module follows.
}
\value{
The value of \code{.morie_ontario_ckan_dump_csv}.
}
\keyword{internal}
