% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{total_compute}
\alias{total_compute}
\title{total_compute}
\usage{
total_compute(samples_seen, model_params) 
}
\arguments{
\item{samples_seen}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{model_params}{Coerced to numeric by the body, with \code{as.numeric}.}
}
\description{
Part of the opnclp_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{compute}, \code{samples_seen}, \code{params}, \code{gmac_scale}.
}
\keyword{internal}
\examples{
total_compute(samples_seen = 5L, model_params = 5L)
}
