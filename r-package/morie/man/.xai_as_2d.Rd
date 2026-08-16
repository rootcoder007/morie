% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.xai_as_2d}
\alias{.xai_as_2d}
\title{.xai_as_2d}
\usage{
.xai_as_2d(X)
}
\arguments{
\item{X}{A matrix; passed to \code{as.matrix}.}
}
\description{
A step of the fairness_xai implementation. Called by \code{morie_fairness_xai_ale}, \code{morie_fairness_xai_ceteris_paribus}, \code{morie_fairness_xai_partial_dependence} and 2 others in the module. See the file header for the source the module follows.
}
\value{
The value of \code{m}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.xai_as_2d(V)
}
