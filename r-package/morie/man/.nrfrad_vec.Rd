% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.nrfrad_vec}
\alias{.nrfrad_vec}
\title{.nrfrad_vec}
\usage{
.nrfrad_vec(x)
}
\arguments{
\item{x}{Optional; may be \code{NULL}. A list; the body reads \code{$sigma} from it.}
}
\description{
Part of the nrfrad_native implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{as.numeric}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.nrfrad_vec(V)
}
