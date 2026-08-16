% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.weighted_prop}
\alias{.weighted_prop}
\title{.weighted_prop}
\usage{
.weighted_prop(x, w)
}
\arguments{
\item{x}{Numeric; combined arithmetically in the body.}
\item{w}{Numeric; passed to \code{sum}.}
}
\description{
A step of the ipw implementation. Called by \code{morie_run_propensity_ipw_analysis}. See the file header for the source the module follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.weighted_prop(V, V)
}
