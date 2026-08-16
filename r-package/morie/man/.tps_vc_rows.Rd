% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.tps_vc_rows}
\alias{.tps_vc_rows}
\title{.tps_vc_rows}
\usage{
.tps_vc_rows(x, top = 20L)
}
\arguments{
\item{x}{Passed to \code{table}.}
\item{top}{A count; the body uses it as \code{seq_len(...)}. Defaults to \code{20L}.}
}
\description{
Part of the tps_all_analyze implementation; see the file header for the source it follows.
}
\value{
The value of \code{lapply}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.tps_vc_rows(V)
}
