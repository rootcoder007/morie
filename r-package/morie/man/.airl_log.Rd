% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.airl_log}
\alias{.airl_log}
\title{.airl_log}
\usage{
.airl_log(x, floor = 1e-300)
}
\arguments{
\item{x}{Numeric; passed to \code{max}.}
\item{floor}{Numeric; passed to \code{max}. Defaults to \code{1e-300}.}
}
\description{
A step of the airl_native implementation. Called by \code{airl}. See the file header for the source the module follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.airl_log(V)
}
