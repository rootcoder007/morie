% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.tbl_to_format}
\alias{.tbl_to_format}
\title{.tbl_to_format}
\usage{
.tbl_to_format(df, fmt = c("dataframe", "latex", "html", "markdown", "text", "csv"), title = "", footnotes = "")
}
\arguments{
\item{df}{Passed to \code{return}.}
\item{fmt}{One of \code{"csv"}, \code{"dataframe"}.}
\item{title}{Passed to \code{nzchar}. Defaults to \code{""}.}
\item{footnotes}{Passed to \code{nzchar}. Defaults to \code{""}.}
}
\description{
Part of the tables_pub implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.tbl_to_format(V)
}
