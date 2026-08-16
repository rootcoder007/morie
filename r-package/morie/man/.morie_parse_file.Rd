% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_parse_file}
\alias{.morie_parse_file}
\title{Parse a downloaded local file according to a known format}
\usage{
.morie_parse_file(path, format, simplify, ...)
}
\arguments{
\item{path}{Passed to \code{.morie_from_json}.}
\item{format}{One of \code{"csv"}, \code{"html"}, \code{"json"}, \code{"tsv"}, \code{"xlsx"}, \code{"xml"}.}
\item{simplify}{A flag; the body branches on it.}
\item{...}{Passed through.}
}
\description{
A step of the data_access implementation. Called by \code{morie_fetch}. See the file header for the source the module follows.
}
\value{
Nothing; this branch always raises.
}
\keyword{internal}
