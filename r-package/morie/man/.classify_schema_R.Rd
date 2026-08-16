% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.classify_schema_R}
\alias{.classify_schema_R}
\title{Internal: turn a dataset_dictionary-style DatasetSchema (a named}
\usage{
.classify_schema_R(schema, dataset_name)
}
\arguments{
\item{schema}{A list; the body reads \code{$columns} from it.}
\item{dataset_name}{Passed to \code{morie_classify_variable}.}
}
\description{
list with $columns) into a list of taxonomies.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
