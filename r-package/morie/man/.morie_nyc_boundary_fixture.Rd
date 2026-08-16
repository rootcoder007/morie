% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_nyc_boundary_fixture}
\alias{.morie_nyc_boundary_fixture}
\title{.morie_nyc_boundary_fixture}
\usage{
.morie_nyc_boundary_fixture(fname, expected_rows = NULL)
}
\arguments{
\item{fname}{Passed to \code{system.file}.}
\item{expected_rows}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
}
\description{
A step of the datasets_nyc_nypd implementation. Called by \code{morie_datasets_nyc_community_districts}, \code{morie_datasets_nyc_council_districts}, \code{morie_datasets_nyc_ntas_2020} and 1 others in the module. See the file header for the source the module follows.
}
\value{
The value of \code{df}, as built in the body.
}
\keyword{internal}
