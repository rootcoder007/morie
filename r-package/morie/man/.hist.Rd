% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.hist}
\alias{.hist}
\title{.hist}
\usage{
.hist(obj, allow_none = FALSE)
}
\arguments{
\item{obj}{Optional; may be \code{NULL}. A vector; its length is taken and its elements indexed.}
\item{allow_none}{A flag; the body branches on it. Defaults to \code{FALSE}.}
}
\description{
Part of the shdsmw_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{list}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.hist(V)
}
