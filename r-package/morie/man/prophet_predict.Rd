% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{prophet_predict}
\alias{prophet_predict}
\alias{morie_prphet_predict}
\title{Forecast at new times, reusing the fitted coefficients}
\usage{
prophet_predict(fit, t_new, seasonalities = NULL, holidays = NULL, holiday_window = c(0, 
    0)) 

morie_prphet_predict(fit, t_new, seasonalities = NULL, holidays = NULL, holiday_window = c(0, 
    0)) 
}
\arguments{
\item{fit}{A fit object as returned by \code{\link{morie_prphet}}.}
\item{t_new}{Numeric vector of new times.}
\item{seasonalities}{Same as for the fit.}
\item{holidays}{Same as for the fit.}
\item{holiday_window}{Same as for the fit.}
}
\description{
Forecast at new times, reusing the fitted coefficients
}
\value{
Numeric vector of forecasts.
}
\keyword{internal}
