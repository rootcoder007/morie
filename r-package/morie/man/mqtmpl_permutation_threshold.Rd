% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mqtmpl_permutation_threshold}
\alias{mqtmpl_permutation_threshold}
\title{mqtmpl_permutation_threshold}
\usage{
mqtmpl_permutation_threshold(y, markers, positions, n_perm = 100, alpha = 0.05, method = "em", step = 0.05, seed = 0, ...)
}
\arguments{
\item{y}{See Usage.}
\item{markers}{See Usage.}
\item{positions}{See Usage.}
\item{n_perm}{Defaults to \code{100}.}
\item{alpha}{Defaults to \code{0.05}.}
\item{method}{Defaults to \code{"em"}.}
\item{step}{Defaults to \code{0.05}.}
\item{seed}{Defaults to \code{0}.}
\item{...}{Passed through.}
}
\description{
Part of the mqtmpl_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{threshold}, \code{alpha}, \code{n_perm}, \code{null_maxima}, \code{median_null}, \code{method}.
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
