% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.wald_ci}
\alias{.wald_ci}
\title{.wald_ci}
\usage{
.wald_ci(estimate, se)
}
\arguments{
\item{estimate}{Numeric; combined arithmetically in the body.}
\item{se}{Numeric; combined arithmetically in the body.}
}
\description{
Part of the study_core implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{c}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.wald_ci(V, V)
}
