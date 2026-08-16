% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{prophet_design}
\alias{prophet_design}
\title{Stack trend, seasonality and holiday columns into one design}
\usage{
prophet_design(t, cps, seasonalities = NULL, holidays = NULL, holiday_window = c(0, 0))
}
\arguments{
\item{t}{Numeric vector of times.}
\item{cps}{Numeric vector of changepoints.}
\item{seasonalities}{Optional list of \code{c(name, period, order)}.}
\item{holidays}{Optional named list of dates per holiday.}
\item{holiday_window}{\code{c(lower, upper)} window around each date.}
}
\description{
Stack trend, seasonality and holiday columns into one design
}
\value{
A list with \code{X} (list of rows), \code{cols}, \code{hn}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
prophet_design(V, V)
}
