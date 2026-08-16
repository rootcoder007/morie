% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.as_numeric_vec}
\alias{.as_numeric_vec}
\title{Accept either a numeric vector or a list of numerics, mirroring}
\usage{
.as_numeric_vec(p)
}
\arguments{
\item{p}{Optional; may be \code{NULL}. A list; the body checks with \code{is.list}.}
}
\description{
k.vec() which iterates over the input.
}
\value{
A vector, from \code{as.numeric}.
}
\keyword{internal}
