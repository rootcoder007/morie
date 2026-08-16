% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_prphet}
\alias{morie_prphet}
\title{Fit a Prophet model by penalised least squares}
\usage{
morie_prphet(t, y, n_changepoints = 10L, changepoint_range = 0.8, changepoints = NULL, seasonalities = NULL, holidays = NULL, holiday_window = c(0, 0), changepoint_prior = 0.05, ridge = 1e-08)
}
\arguments{
\item{t}{Numeric vector of times.}
\item{y}{Numeric vector of observations.}
\item{n_changepoints}{Integer number of candidate changepoints.}
\item{changepoint_range}{Fraction of the time range to span.}
\item{changepoints}{Optional explicit vector of changepoints.}
\item{seasonalities}{Optional list of \code{c(name, period, order)}.}
\item{holidays}{Optional named list of dates per holiday.}
\item{holiday_window}{\code{c(lower, upper)} window around each date.}
\item{changepoint_prior}{Positive Laplace scale tau.}
\item{ridge}{Small L2 stabiliser on every coefficient.}
}
\description{
Soft-thresholded cyclic coordinate descent. The Laplace prior on the deltas is an L1 penalty of \code{1/changepoint_prior} on the rate adjustments only -- never on k, m, the seasonal coefficients or the holiday indicators.
}
\value{
A list with \code{estimate}, \code{fitted}, \code{residual},   \code{coef}, \code{beta}, \code{columns}, \code{changepoints},   \code{deltas}, \code{k}, \code{m}, \code{trend}, \code{holiday_names},   \code{t}, \code{n}, \code{changepoint_prior},   \code{n_active_changepoints}, \code{sigma}, \code{seasonalities},   \code{method}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
morie_prphet(V, V)
}
