% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.seckdf_as_bytes}
\alias{.seckdf_as_bytes}
\title{.seckdf_as_bytes}
\usage{
.seckdf_as_bytes(x)
}
\arguments{
\item{x}{Optional; may be \code{NULL}. Character; the body checks with \code{is.character}.}
}
\description{
Part of the seckdf_native implementation; see the file header for the source it follows.
}
\value{
Nothing; this branch always raises.
}
\keyword{internal}
\examples{
S <- c("a", "b", "c")
.seckdf_as_bytes(S)
}
