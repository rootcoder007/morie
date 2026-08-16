% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.tbl_fmt_num}
\alias{.tbl_fmt_num}
\title{.tbl_fmt_num}
\usage{
.tbl_fmt_num(x, digits = 2L, apa = FALSE)
}
\arguments{
\item{x}{Numeric; passed to \code{abs}.}
\item{digits}{Defaults to \code{2L}.}
\item{apa}{A flag; the body branches on it. Defaults to \code{FALSE}.}
}
\description{
Part of the tables_pub implementation; see the file header for the source it follows.
}
\value{
The value of \code{s}, as built in the body.
}
\keyword{internal}
\examples{
.tbl_fmt_num(x = 5L)
}
