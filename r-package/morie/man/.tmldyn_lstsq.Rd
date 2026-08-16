% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.tmldyn_lstsq}
\alias{.tmldyn_lstsq}
\title{.tmldyn_lstsq}
\usage{
.tmldyn_lstsq(X, yv, ridge = 1e-08)
}
\arguments{
\item{X}{See Usage.}
\item{yv}{A matrix; passed to \code{crossprod}.}
\item{ridge}{Numeric; combined arithmetically in the body. Defaults to \code{1e-08}.}
}
\description{
Part of the tmldyn_native implementation; see the file header for the source it follows.
}
\value{
A matrix, from \code{solve}.
}
\keyword{internal}
