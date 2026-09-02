% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_sarima}
\alias{morie_sarima}
\title{morie_sarima}
\usage{
morie_sarima(y, order = c(0, 1, 1), seasonal_order = c(0, 1, 1), s = 12, method = "ml", 
    start = NULL) 
}
\arguments{
\item{y}{Passed to \code{.sarima_fit}.}
\item{order}{Passed to \code{.sarima_fit}. Defaults to \code{c(0, 1, 1)}.}
\item{seasonal_order}{Passed to \code{.sarima_fit}. Defaults to \code{c(0, 1, 1)}.}
\item{s}{Passed to \code{.sarima_fit}. Defaults to \code{12}.}
\item{method}{Passed to \code{.sarima_fit}. Defaults to \code{"ml"}.}
\item{start}{Passed to \code{.sarima_fit}.}
}
\description{
Part of the sarima_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{.sarima_fit}.
}
\keyword{internal}
