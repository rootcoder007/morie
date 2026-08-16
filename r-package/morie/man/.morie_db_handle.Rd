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
\item{db_path}{Defaults to \code{NULL}.}
}
\description{
Part of the database implementation; see the file header for the source it follows.
}
\value{
A list with \code{con}, \code{close}.
}
\keyword{internal}
