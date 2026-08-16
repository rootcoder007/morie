% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_cdaeRC_decode}
\alias{morie_cdaeRC_decode}
\title{morie_cdaeRC_decode}
\usage{
morie_cdaeRC_decode(z, Wp, bp, items = NULL, activation = "sigmoid")
}
\arguments{
\item{z}{A vector; its length is taken and its elements indexed.}
\item{Wp}{A matrix; indexed by row and column.}
\item{bp}{A vector; its length is taken and its elements indexed.}
\item{items}{Defaults to \code{NULL}.}
\item{activation}{Passed to \code{.cdae_act}. Defaults to \code{"sigmoid"}.}
}
\description{
A step of the cdaeRC_native implementation. No other function in the package calls it. See the file header for the source the module follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
