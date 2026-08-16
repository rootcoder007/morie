% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.run_data_wrangling_module_internal}
\alias{.run_data_wrangling_module_internal}
\title{.run_data_wrangling_module_internal}
\usage{
.run_data_wrangling_module_internal(data, cpads_csv = NULL, output_dir = NULL)
}
\arguments{
\item{data}{A matrix; passed to \code{nrow}.}
\item{cpads_csv}{Optional; may be \code{NULL}. Passed to \code{.resolve_cpads_csv}.}
\item{output_dir}{Defaults to \code{NULL}.}
}
\description{
Part of the study_core implementation; see the file header for the source it follows.
}
\value{
A list with \code{data_na_summary}, \code{data_wrangling_log}.
}
\keyword{internal}
