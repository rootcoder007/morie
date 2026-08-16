% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.binary_power_required_n}
\alias{.binary_power_required_n}
\title{.binary_power_required_n}
\usage{
.binary_power_required_n(p1, p2, alpha = 0.05, power = 0.8)
}
\arguments{
\item{p1}{Numeric; passed to \code{sqrt}.}
\item{p2}{Numeric; passed to \code{sqrt}.}
\item{alpha}{Numeric; combined arithmetically in the body. Defaults to \code{0.05}.}
\item{power}{Defaults to \code{0.8}.}
}
\description{
Part of the study_reporting implementation; see the file header for the source it follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
.binary_power_required_n(p1 = 5L, p2 = 5L)
}
