% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.level_from_spec}
\alias{.level_from_spec}
\title{.level_from_spec}
\usage{
.level_from_spec(col_name, dtype, valid_values, dataset_name)
}
\arguments{
\item{col_name}{Character; passed to \code{grepl}.}
\item{dtype}{One of \code{"bool"}, \code{"date"}, \code{"datetime"}, \code{"float"}, \code{"int"}, \code{"string"}.}
\item{valid_values}{Optional; may be \code{NULL}. A vector; its length is taken.}
\item{dataset_name}{Accepted by the signature and not used anywhere in the body.}
}
\description{
Part of the variable_taxonomy implementation; see the file header for the source it follows.
}
\value{
A character value.
}
\keyword{internal}
