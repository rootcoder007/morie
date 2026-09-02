% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{utility}
\alias{utility}
\title{Utility = efficacy per person-day in quarantine (eq. 4)}
\usage{
utility(efficacy, days_in_quarantine) 
}
\arguments{
\item{efficacy}{Numeric efficacy.}
\item{days_in_quarantine}{Positive number of days.}
}
\description{
Utility = efficacy per person-day in quarantine (eq. 4)
}
\value{
Numeric.
}
\keyword{internal}
\examples{
utility(efficacy = c(1, 2, 3, 4, 5, 6, 7, 8), days_in_quarantine = 5L)
}
