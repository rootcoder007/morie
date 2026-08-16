% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.ghlgd_negll}
\alias{.ghlgd_negll}
\title{Internal: penalised log-spline density negative log-likelihood}
\usage{
.ghlgd_negll(theta, Bx, Bg, gz, n)
}
\arguments{
\item{theta}{A matrix; passed to \code{\%*\%}.}
\item{Bx}{A matrix; passed to \code{\%*\%}.}
\item{Bg}{A matrix; passed to \code{\%*\%}.}
\item{gz}{See Usage.}
\item{n}{Numeric; combined arithmetically in the body.}
}
\description{
Extracted from the morie_ghosal_log_density() optimiser closure for direct unit-testing. `Bx`/`Bg` are the data and grid basis matrices, `gz` the standardised evaluation grid, `n` the sample size.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
.ghlgd_negll(theta = c(1, 2, 3, 4, 5, 6, 7, 8), Bx = c(1, 2, 3, 4, 5, 6, 7, 8), Bg = c(1, 2, 3, 4, 5, 6, 7, 8), gz = c(1, 2, 3, 4, 5, 6, 7, 8), n = 5L)
}
