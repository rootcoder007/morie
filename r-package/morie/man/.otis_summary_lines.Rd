% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.otis_summary_lines}
\alias{.otis_summary_lines}
\title{.otis_summary_lines}
\usage{
.otis_summary_lines(df, ds_id, description = NULL, series = NULL, primary_metric = NULL)
}
\arguments{
\item{df}{A matrix; passed to \code{nrow}.}
\item{ds_id}{Passed to \code{sprintf}.}
\item{description}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
\item{series}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
\item{primary_metric}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
}
\description{
Part of the otis_all_analyze implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.otis_summary_lines(V, V)
}
