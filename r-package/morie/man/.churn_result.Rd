% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.churn_result}
\alias{.churn_result}
\title{Result constructor shared with otis.R (.otis_result lives there but}
\usage{
.churn_result(title, summary_lines = list(), tables = list(), interpretation = "", warnings = character(0), payload = list())
}
\arguments{
\item{title}{See Usage.}
\item{summary_lines}{Defaults to \code{list()}.}
\item{tables}{Defaults to \code{list()}.}
\item{interpretation}{Defaults to \code{""}.}
\item{warnings}{Defaults to \code{character(0)}.}
\item{payload}{Defaults to \code{list()}.}
}
\description{
we define a thin alias here to avoid load-order coupling). We name it `.churn_result` so it can co-exist if file load order changes.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
