% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{fourier_terms}
\alias{fourier_terms}
\title{Cosine and sine Fourier pairs, exactly periodic with \code{period}}
\usage{
fourier_terms(t, period, order)
}
\arguments{
\item{t}{Numeric vector of times.}
\item{period}{Positive period.}
\item{order}{Integer, number of harmonics.}
}
\description{
Cosine and sine Fourier pairs, exactly periodic with \code{period}
}
\value{
A list of rows, each \code{2 * order} long.
}
\keyword{internal}
