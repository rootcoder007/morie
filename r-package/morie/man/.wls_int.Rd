% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.wls_int}
\alias{.wls_int}
\title{.wls_int}
\usage{
.wls_int(Xm, yv, w, ridge)
}
\arguments{
\item{Xm}{A matrix; passed to \code{nrow}.}
\item{yv}{A matrix; passed to \code{\%*\%}.}
\item{w}{A matrix; passed to \code{diag}.}
\item{ridge}{Numeric; combined arithmetically in the body.}
}
\description{
Part of the tmldyk_native implementation; see the file header for the source it follows.
}
\value{
A matrix, from \code{solve}.
}
\keyword{internal}
