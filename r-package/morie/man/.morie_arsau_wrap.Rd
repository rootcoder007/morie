% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_arsau_wrap}
\alias{.morie_arsau_wrap}
\title{.morie_arsau_wrap}
\usage{
.morie_arsau_wrap(title, call, sub_results, data, sidecar, year_or_range, kind, language, is_valid, extra_interpretation = "")
}
\arguments{
\item{title}{Carried through into a list the body builds.}
\item{call}{Carried through into a list the body builds.}
\item{sub_results}{A vector; its length is taken and its elements indexed.}
\item{data}{Optional; may be \code{NULL}. A matrix; passed to \code{nrow}.}
\item{sidecar}{Carried through into a list the body builds.}
\item{year_or_range}{Carried through into a list the body builds.}
\item{kind}{Carried through into a list the body builds.}
\item{language}{Carried through into a list the body builds.}
\item{is_valid}{A flag; the body branches on it.}
\item{extra_interpretation}{Passed to \code{paste}. Defaults to \code{""}.}
}
\description{
A step of the arsau_analyze implementation. Called by \code{morie_arsau_analyze_aggregate_summary}, \code{morie_arsau_analyze_detailed_dataset}, \code{morie_arsau_analyze_individual_records} and 3 others in the module. See the file header for the source the module follows.
}
\value{
The value of \code{payload}, as built in the body.
}
\keyword{internal}
