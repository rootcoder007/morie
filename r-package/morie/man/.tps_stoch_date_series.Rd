% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.tps_stoch_date_series}
\alias{.tps_stoch_date_series}
\title{Return a numeric vector of incident times (POSIXct)}
\usage{
.tps_stoch_date_series(df, min_year = 2014L)
}
\arguments{
\item{df}{A list; the body reads \code{$OCC_DAY}, \code{$OCC_MONTH}, \code{$OCC_YEAR} from it.}
\item{min_year}{Defaults to \code{2014L}.}
}
\description{
Prefer integer-triple OCC_YEAR/OCC_MONTH/OCC_DAY (local-time decomposition unaffected by ArcGIS UTC conversion); fall back to OCC_DATE / REPORT_DATE.
}
\value{
A vector, from \code{sort}.
}
\keyword{internal}
