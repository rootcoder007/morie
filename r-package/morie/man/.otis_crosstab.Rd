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
\item{df}{A matrix; indexed by row and column.}
\item{row}{Passed to \code{c}.}
\item{col}{Passed to \code{c}.}
\item{value}{Passed to \code{c}.}
\item{aggfunc}{Compared against \code{"max"}. Defaults to \code{c("sum", "max")}.}
\item{top_rows}{Numeric; passed to \code{min}. Defaults to \code{20L}.}
}
\description{
Part of the otis_all_analyze implementation; see the file header for the source it follows.
}
\value{
A list with \code{title}, \code{headers}, \code{rows}.
}
\keyword{internal}
\examples{
.otis_crosstab(df = data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9)), row = c(1, 2, 3, 4, 5, 6, 7, 8), col = c("a", "b", "c"), value = c(1, 2, 3, 4, 5, 6, 7, 8))
}
