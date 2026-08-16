% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.tps_temporal_monthly}
\alias{.tps_temporal_monthly}
\title{Build monthly counts from an arbitrary date column. Returns a list}
\usage{
.tps_temporal_monthly(df)
}
\arguments{
\item{df}{A list; the body reads \code{$OCC_DAY}, \code{$OCC_MONTH}, \code{$OCC_YEAR} from it.}
}
\description{
with $dates (POSIXct, first-of-month) and $counts (integer).
}
\value{
A list with \code{dates}, \code{counts}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.tps_temporal_monthly(V)
}
