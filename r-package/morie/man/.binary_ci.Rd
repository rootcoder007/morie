% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.binary_ci}
\alias{.binary_ci}
\title{.binary_ci}
\usage{
.binary_ci(successes, n)
}
\arguments{
\item{successes}{Passed to \code{.safe_divide}.}
\item{n}{Numeric; passed to \code{max}.}
}
\description{
Part of the study_core implementation; see the file header for the source it follows.
}
\value{
A list with \code{p}, \code{se}, \code{ci}.
}
\keyword{internal}
\examples{
.binary_ci(successes = c(1, 2, 3, 4, 5, 6, 7, 8), n = 5L)
}
