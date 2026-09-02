% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{nbeats_seasonality_basis}
\alias{nbeats_seasonality_basis}
\title{nbeats_seasonality_basis}
\usage{
nbeats_seasonality_basis(length, harmonics, offset = 0, period = NULL) 
}
\arguments{
\item{length}{A count; the body uses it as \code{seq_len(...)}.}
\item{harmonics}{A count; the body uses it as \code{seq_len(...)}.}
\item{offset}{Numeric; combined arithmetically in the body. Defaults to \code{0}.}
\item{period}{Optional; may be \code{NULL}. Coerced to numeric by the body, with \code{as.numeric}.}
}
\description{
Part of the nbeats_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{do.call}.
}
\keyword{internal}
\examples{
nbeats_seasonality_basis(length = 5L, harmonics = 5L)
}
