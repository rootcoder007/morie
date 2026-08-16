% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.tps_stoch_result}
\alias{.tps_stoch_result}
\title{.tps_stoch_result}
\usage{
.tps_stoch_result(title, call, summary_lines = list(), warnings = character(0), interpretation = "", ...)
}
\arguments{
\item{title}{Carried through into a list the body builds.}
\item{call}{Carried through into a list the body builds.}
\item{summary_lines}{Carried through into a list the body builds. Defaults to \code{list()}.}
\item{warnings}{Carried through into a list the body builds. Defaults to \code{character(0)}.}
\item{interpretation}{Carried through into a list the body builds. Defaults to \code{""}.}
\item{...}{Passed through.}
}
\description{
Part of the tps_stochastic implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.tps_stoch_result(V, V)
}
