% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.gtrf_normalize}
\alias{.gtrf_normalize}
\title{.gtrf_normalize}
\usage{
.gtrf_normalize(X, how)
}
\arguments{
\item{X}{A matrix; indexed by row and column.}
\item{how}{One of \code{"batch"}, \code{"none"}.}
}
\description{
A step of the gtrf_native implementation. Called by \code{morie_gtrf_layer}. See the file header for the source the module follows.
}
\value{
One of two values, depending on the branch taken.
}
\keyword{internal}
