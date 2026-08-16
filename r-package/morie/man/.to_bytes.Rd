% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.to_bytes}
\alias{.to_bytes}
\title{.to_bytes}
\usage{
.to_bytes(ws)
}
\arguments{
\item{ws}{Iterated over elementwise, with \code{lapply}.}
}
\description{
Part of the secarg_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{do.call}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.to_bytes(V)
}
