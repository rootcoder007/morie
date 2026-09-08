% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{nbeats_trend_basis}
\alias{nbeats_trend_basis}
\title{nbeats_trend_basis}
\usage{
nbeats_trend_basis(length, degree, offset = 0, scale = NULL) 
}
\arguments{
\item{length}{A count; the body uses it as \code{seq_len(...)}.}
\item{degree}{Numeric; combined arithmetically in the body.}
\item{offset}{Numeric; combined arithmetically in the body. Defaults to \code{0}.}
\item{scale}{Optional; may be \code{NULL}. Coerced to numeric by the body, with \code{as.numeric}.}
}
\description{
Part of the nbeats_native implementation; see the file header for the source it follows.
}
\value{
A matrix, from \code{t}.
}
\keyword{internal}
\examples{
nbeats_trend_basis(length = 5L, degree = 5L)
}
