% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.otis_contingency_chi2}
\alias{.otis_contingency_chi2}
\title{.otis_contingency_chi2}
\usage{
.otis_contingency_chi2(df, row, col, value)
}
\arguments{
\item{df}{A matrix; indexed by row and column.}
\item{row}{Passed to \code{c}.}
\item{col}{Passed to \code{c}.}
\item{value}{Passed to \code{c}.}
}
\description{
Part of the otis_all_analyze implementation; see the file header for the source it follows.
}
\value{
A list with \code{table}, \code{stats}.
}
\keyword{internal}
\examples{
.otis_contingency_chi2(df = data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9)), row = c(1, 2, 3, 4, 5, 6, 7, 8), col = c("a", "b", "c"), value = c(1, 2, 3, 4, 5, 6, 7, 8))
}
