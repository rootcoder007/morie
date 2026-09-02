% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{holiday_matrix}
\alias{holiday_matrix}
\alias{morie_prphet_holiday_matrix}
\title{One indicator per holiday, optionally widened by a window}
\usage{
holiday_matrix(t, holidays, lower = 0, upper = 0) 

morie_prphet_holiday_matrix(t, holidays, lower = 0, upper = 0) 
}
\arguments{
\item{t}{Numeric vector of times.}
\item{holidays}{Named list of dates per holiday.}
\item{lower}{Integer, days before the holiday to flag.}
\item{upper}{Integer, days after the holiday to flag.}
}
\description{
One indicator per holiday, optionally widened by a window
}
\value{
A list with \code{matrix} (list of rows) and \code{names}   (sorted holiday names).
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
holiday_matrix(V, V)
}
