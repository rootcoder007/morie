% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.otis_year_trend}
\alias{.otis_year_trend}
\title{.otis_year_trend}
\usage{
.otis_year_trend(df, value, year_col = NULL)
}
\arguments{
\item{df}{A vector; indexed elementwise.}
\item{value}{Passed to \code{\%in\%}.}
\item{year_col}{Passed to \code{\%||\%}.}
}
\description{
Part of the otis_all_analyze implementation; see the file header for the source it follows.
}
\value{
A list with \code{title}, \code{headers}, \code{rows}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.otis_year_trend(V, V)
}
