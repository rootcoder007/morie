% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.otis_c_simple}
\alias{.otis_c_simple}
\title{.otis_c_simple}
\usage{
.otis_c_simple(ds_id, description, row_col, col_col = "Region_MostRecentPlacement", value_col = "NumberIndividuals_RestrictiveConfinement")
}
\arguments{
\item{ds_id}{Passed to \code{.otis_summary_lines}.}
\item{description}{Passed to \code{.otis_summary_lines}.}
\item{row_col}{Passed to \code{.otis_crosstab}.}
\item{col_col}{Passed to \code{.otis_crosstab}.}
\item{value_col}{Passed to \code{.otis_crosstab}.}
}
\description{
Part of the otis_all_analyze implementation; see the file header for the source it follows.
}
\value{
The value of \code{function}.
}
\keyword{internal}
\examples{
.otis_c_simple(ds_id = c(1, 2, 3, 4, 5, 6, 7, 8), description = c(1, 2, 3, 4, 5, 6, 7, 8), row_col = c(1, 2, 3, 4, 5, 6, 7, 8))
}
