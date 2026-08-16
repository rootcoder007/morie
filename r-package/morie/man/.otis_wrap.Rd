% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.otis_wrap}
\alias{.otis_wrap}
\title{.otis_wrap}
\usage{
.otis_wrap(title, summary_lines, tables = list(), interpretation = NULL, warnings = character(0), payload = NULL)
}
\arguments{
\item{title}{Carried through into a list the body builds.}
\item{summary_lines}{Carried through into a list the body builds.}
\item{tables}{Iterated over elementwise, with \code{Filter}. Defaults to \code{list()}.}
\item{interpretation}{Passed to \code{\%||\%}.}
\item{warnings}{Carried through into a list the body builds. Defaults to \code{character(0)}.}
\item{payload}{Carried through into a list the body builds.}
}
\description{
Part of the otis_all_analyze implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.otis_wrap(V, V)
}
