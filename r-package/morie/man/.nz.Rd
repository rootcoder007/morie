% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.nz}
\alias{.nz}
\title{Small helper: first non-empty scalar, else ""}
\usage{
.nz(...)
}
\arguments{
\item{...}{Passed through.}
}
\description{
A step of the data_access implementation. Called by \code{morie_ckan_search}, \code{morie_fetch_arcgis}. See the file header for the source the module follows.
}
\value{
A character value.
}
\keyword{internal}
\examples{
.nz()
}
