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
\item{theta}{See Usage.}
\item{Bx}{See Usage.}
\item{Bg}{See Usage.}
\item{gz}{See Usage.}
\item{n}{See Usage.}
}
\description{
Extracted from the morie_ghosal_log_density() optimiser closure for direct unit-testing. `Bx`/`Bg` are the data and grid basis matrices, `gz` the standardised evaluation grid, `n` the sample size.
}
\value{
A numeric value.
}
\keyword{internal}
