% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{instance_norm}
\alias{instance_norm}
\title{Standardise one series to zero mean and unit variance. Applied}
\usage{
instance_norm(x)
}
\arguments{
\item{x}{See Usage.}
}
\description{
per instance before patching and reversed afterwards.
}
\value{
A list with \code{normalised}, \code{mean}, \code{sd}, \code{degenerate}.
}
\keyword{internal}
