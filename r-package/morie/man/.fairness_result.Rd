% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.fairness_result}
\alias{.fairness_result}
\title{.fairness_result}
\usage{
.fairness_result(title, call, summary_lines = list(), warnings = character(0), interpretation = "", ...)
}
\arguments{
\item{title}{See Usage.}
\item{call}{See Usage.}
\item{summary_lines}{Defaults to \code{list()}.}
\item{warnings}{Defaults to \code{character(0)}.}
\item{interpretation}{Defaults to \code{""}.}
\item{...}{Passed through.}
}
\description{
A step of the fairness_gan implementation. Called by \code{.fairness_no_backend_result}, \code{morie_fairness_ctgan_debiaser}, \code{morie_fairness_spatial_gan}. See the file header for the source the module follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
