% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.changepoints}
\alias{.changepoints}
\title{Place candidate changepoints}
\usage{
.changepoints(t, n_changepoints, changepoint_range = 0.8, changepoints = NULL)
}
\arguments{
\item{t}{Numeric vector of times.}
\item{n_changepoints}{Integer, number of candidate changepoints.}
\item{changepoint_range}{Fraction of the time range to span.}
\item{changepoints}{Optional explicit vector of changepoints.}
}
\description{
If \code{changepoints} is supplied it is used as-is; otherwise \code{n_changepoints} equally spaced points are placed over the first \code{changepoint_range} of the time range.
}
\value{
Numeric vector of changepoints.
}
\keyword{internal}
\examples{
.changepoints(t = c(1, 2, 3, 4, 5, 6, 7, 8), n_changepoints = 5L)
}
