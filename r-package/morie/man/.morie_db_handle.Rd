% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_db_handle}
\alias{.morie_db_handle}
\title{Returns: list(con = DBIConnection, close = logical)}
\usage{
.morie_db_handle(con = NULL, db_path = NULL)
}
\arguments{
\item{con}{Defaults to \code{NULL}.}
\item{db_path}{Passed to \code{morie_db_connect}.}
}
\description{
A step of the database implementation. Called by \code{morie_cache_list}, \code{morie_cache_load}, \code{morie_cache_store}. See the file header for the source the module follows.
}
\value{
A list with \code{con}, \code{close}.
}
\keyword{internal}
