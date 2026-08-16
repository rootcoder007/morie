% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{piecewise_trend}
\alias{piecewise_trend}
\title{Evaluate the piecewise-linear trend (Eq. 4 of Taylor & Letham 2018)}
\usage{
piecewise_trend(t, k_rate, m_off, deltas, cps)
}
\arguments{
\item{t}{Numeric vector of times.}
\item{k_rate}{Initial rate.}
\item{m_off}{Initial offset.}
\item{deltas}{Numeric vector of rate adjustments, one per changepoint.}
\item{cps}{Numeric vector of changepoints.}
}
\description{
The offsets carry \code{gamma_j = -s_j delta_j} so the segments join by construction.
}
\value{
Numeric vector of trend values.
}
\keyword{internal}
\examples{
piecewise_trend(t = c(1, 2, 3, 4, 5, 6, 7, 8), k_rate = c(1, 2, 3, 4, 5, 6, 7, 8), m_off = c(1, 2, 3, 4, 5, 6, 7, 8), deltas = c(1, 2, 3, 4, 5, 6, 7, 8), cps = c(1, 2, 3, 4, 5, 6, 7, 8))
}
