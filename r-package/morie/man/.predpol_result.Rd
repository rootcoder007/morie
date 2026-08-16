% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.predpol_result}
\alias{.predpol_result}
\title{.predpol_result}
\usage{
.predpol_result(title, call, summary_lines = list(), warnings = character(0), interpretation = "", ...)
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
A step of the fairness_predpol implementation. Called by \code{morie_fairness_predpol_calibration_audit}, \code{morie_fairness_predpol_score_disparity}. See the file header for the source the module follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.predpol_result(V, V)
}
