% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.otis_crosstab}
\alias{.otis_crosstab}
\title{.otis_crosstab}
\usage{
.otis_crosstab(df, row, col, value, aggfunc = c("sum", "max"), top_rows = 20L)
}
\arguments{
\item{df}{See Usage.}
\item{row}{See Usage.}
\item{col}{See Usage.}
\item{value}{See Usage.}
\item{aggfunc}{Defaults to \code{c("sum", "max")}.}
\item{top_rows}{Defaults to \code{20L}.}
}
\description{
Part of the otis_all_analyze implementation; see the file header for the source it follows.
}
\value{
A list with \code{title}, \code{headers}, \code{rows}.
}
\keyword{internal}
