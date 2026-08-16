% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.gp_as_nugget}
\alias{.gp_as_nugget}
\title{.gp_as_nugget}
\usage{
.gp_as_nugget(nugget, n)
}
\arguments{
\item{nugget}{Optional; may be \code{NULL}. Coerced to numeric by the body, with \code{as.numeric}.}
\item{n}{A count; the body uses it as \code{rep(...)}.}
}
\description{
A step of the abcgp_native implementation. Called by \code{gp_fit}. See the file header for the source the module follows.
}
\value{
The value of \code{pmax}.
}
\keyword{internal}
\examples{
.gp_as_nugget(nugget = 5L, n = 5L)
}
