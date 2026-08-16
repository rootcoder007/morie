% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_siu_rich}
\alias{.morie_siu_rich}
\title{.morie_siu_rich}
\usage{
.morie_siu_rich(title, summary_lines = list(), tables = list(), interpretation = "", warnings = character(), payload = list())
}
\arguments{
\item{title}{See Usage.}
\item{summary_lines}{Defaults to \code{list()}.}
\item{tables}{Defaults to \code{list()}.}
\item{interpretation}{Defaults to \code{""}.}
\item{warnings}{Defaults to \code{character()}.}
\item{payload}{Defaults to \code{list()}.}
}
\description{
RichResult-style constructor for the R side --------------------------------------------------------------------------- Returns a named list classed for morie\\\'s rich-output dispatch. Mirrors the structure of morie.fn._richresult.RichResult so that downstream `describe()` / `morie_print_rich()` consumers can render the same multi-section paragraph layout from R.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
