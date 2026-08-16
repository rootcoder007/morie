% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.otis_tps_incidents_by_year}
\alias{.otis_tps_incidents_by_year}
\title{.otis_tps_incidents_by_year}
\usage{
.otis_tps_incidents_by_year(df_tps)
}
\arguments{
\item{df_tps}{A vector; indexed elementwise.}
}
\description{
Part of the otis_tps_overlay implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
\examples{
D <- data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9))
.otis_tps_incidents_by_year(D)
}
