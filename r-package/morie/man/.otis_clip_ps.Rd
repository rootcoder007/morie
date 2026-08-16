% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.otis_clip_ps}
\alias{.otis_clip_ps}
\title{Clip propensity away from 0/1}
\usage{
.otis_clip_ps(e, eps = 0.02)
}
\arguments{
\item{e}{Passed to \code{pmax}.}
\item{eps}{Numeric; combined arithmetically in the body. Defaults to \code{0.02}.}
}
\description{
Part of the otis_causal implementation; see the file header for the source it follows.
}
\value{
The value of \code{pmin}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.otis_clip_ps(V)
}
