% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{neighbours}
\alias{neighbours}
\title{neighbours}
\usage{
neighbours(order, seasonal_order, constant, s)
}
\arguments{
\item{order}{A vector; indexed elementwise.}
\item{seasonal_order}{A vector; indexed elementwise.}
\item{constant}{A flag; the body branches on it.}
\item{s}{Coerced to integer by the body, with \code{as.integer}.}
}
\description{
Part of the sarimax_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
