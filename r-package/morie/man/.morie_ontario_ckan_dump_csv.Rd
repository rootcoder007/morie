% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_ontario_ckan_dump_csv}
\alias{.morie_ontario_ckan_dump_csv}
\title{Internal: GET the Ontario CKAN datastore_dump JSON endpoint for a}
\usage{
.morie_ontario_ckan_dump_csv(resource_id, limit = 200000L)
}
\arguments{
\item{resource_id}{See Usage.}
\item{limit}{Accepted by the signature and not used anywhere in the body. Defaults to \code{200000L}.}
}
\description{
given resource_id; return a parsed data.frame. Mock this in tests.
}
\value{
Nothing; this branch always raises.
}
\keyword{internal}
