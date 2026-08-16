% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.dmlc_ps}
\alias{.dmlc_ps}
\title{Ridge-logistic propensity (tiny ridge for separation), predicted +}
\usage{
.dmlc_ps(Xtr, dtr, Xte, eps)
}
\arguments{
\item{Xtr}{See Usage.}
\item{dtr}{See Usage.}
\item{Xte}{A matrix; passed to \code{\%*\%}.}
\item{eps}{Numeric; combined arithmetically in the body.}
}
\description{
clipped Part of the dml_clustered implementation; see the file header for the source it follows.
}
\value{
The value of \code{pmin}.
}
\keyword{internal}
