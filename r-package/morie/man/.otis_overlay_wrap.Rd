% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.otis_overlay_wrap}
\alias{.otis_overlay_wrap}
\title{Wrap a list as a morie_otis_analysis_result (lazy reference to the}
\usage{
.otis_overlay_wrap(title, summary_lines, tables = list(), interpretation = "", warnings = character(0), payload = NULL)
}
\arguments{
\item{title}{See Usage.}
\item{summary_lines}{See Usage.}
\item{tables}{Iterated over elementwise, with \code{Filter}. Defaults to \code{list()}.}
\item{interpretation}{Defaults to \code{""}.}
\item{warnings}{Defaults to \code{character(0)}.}
\item{payload}{Defaults to \code{NULL}.}
}
\description{
constructor defined in otis_all_analyze.R -- both files ship in the same R/ collation order, so this resolves at package-load time).
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
