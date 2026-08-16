% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.new_diag_report}
\alias{.new_diag_report}
\title{.new_diag_report}
\usage{
.new_diag_report(residuals, influence, collinearity, gof, spec_tests, assessment)
}
\arguments{
\item{residuals}{Carried through into a list the body builds.}
\item{influence}{Carried through into a list the body builds.}
\item{collinearity}{Carried through into a list the body builds.}
\item{gof}{Carried through into a list the body builds.}
\item{spec_tests}{Carried through into a list the body builds.}
\item{assessment}{Carried through into a list the body builds.}
}
\description{
A step of the diagnostics implementation. Called by \code{full_diagnostics}. See the file header for the source the module follows.
}
\value{
The value of \code{structure}.
}
\keyword{internal}
\examples{
.new_diag_report(residuals = c(1, 2, 3, 4, 5, 6, 7, 8), influence = c(1, 2, 3, 4, 5, 6, 7, 8), collinearity = c(1, 2, 3, 4, 5, 6, 7, 8), gof = c(1, 2, 3, 4, 5, 6, 7, 8), spec_tests = c(1, 2, 3, 4, 5, 6, 7, 8), assessment = c(1, 2, 3, 4, 5, 6, 7, 8))
}
