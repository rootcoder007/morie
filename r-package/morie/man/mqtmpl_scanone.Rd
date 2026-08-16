% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mqtmpl_scanone}
\alias{mqtmpl_scanone}
\title{mqtmpl_scanone}
\usage{
mqtmpl_scanone(y, markers, positions, method = "em", step = 0.02, covariates = list(), error_rate = 0)
}
\arguments{
\item{y}{A vector; its length is taken and its elements indexed.}
\item{markers}{A vector; its length is taken and its elements indexed.}
\item{positions}{A vector; indexed elementwise.}
\item{method}{One of \code{"imp"}, \code{"mr"}. Defaults to \code{"em"}.}
\item{step}{Defaults to \code{0.02}.}
\item{covariates}{A vector; its length is taken. Defaults to \code{list()}.}
\item{error_rate}{Defaults to \code{0}.}
}
\description{
Part of the mqtmpl_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{peak_lod}, \code{peak_position}, \code{position}, \code{lod}, \code{method_used}, \code{n_covariates}, \code{error_rate}, \code{method}.
}
\references{
  Broman et al. (2003) "R/qtl" Bioinformatics 19(7), 889-890
  Lander & Botstein (1989) Genetics 121(1), 185-199
  Churchill & Doerge (1994) Genetics 138(3), 963-971
  Baum et al. (1970) Ann. Math. Statist. 41(1), 164-171
  Sen & Churchill (2001) Genetics 159(1), 371-387
Base R only.
}
\keyword{internal}
