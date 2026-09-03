% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{kalman_filter_1d}
\alias{kalman_filter_1d}
\title{Kalman filter for the scalar linear-Gaussian state-space model}
\usage{
kalman_filter_1d(y, a, q, c, r, m0 = 0, p0 = 1) 
}
\arguments{
\item{y}{Numeric vector of observations.}
\item{a}{Process coefficient.}
\item{q}{Process variance.}
\item{c}{Measurement coefficient.}
\item{r}{Measurement variance.}
\item{m0}{Initial state mean.}
\item{p0}{Initial state variance.}
}
\description{
\code{x_n = a x_{n-1} + N(0, q)}, \code{y_n = c x_n + N(0, r)}, starting at \code{(m0, p0)}. Provided so the particle filter can be checked against the exact answer.
}
\value{
A list with \code{filtered_mean} and \code{loglik}.
}
\keyword{internal}
\examples{
kalman_filter_1d(y = c(1, 2, 3, 4, 5, 6, 7, 8), a = c(1, 2, 3, 4, 5, 6, 7, 8), q = 0.5,
  c = c(1, 2, 3, 4, 5, 6, 7, 8), r = c(1, 2, 3, 4, 5, 6, 7, 8))
}
