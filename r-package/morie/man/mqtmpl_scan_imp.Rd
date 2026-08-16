% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mqtmpl_scan_imp}
\alias{mqtmpl_scan_imp}
\title{mqtmpl_scan_imp}
\usage{
mqtmpl_scan_imp(y, markers, positions, step, n_imp, error_rate, seed)
}
\arguments{
\item{y}{A vector; its length is taken.}
\item{markers}{A vector; its length is taken and its elements indexed.}
\item{positions}{A vector; its length is taken and its elements indexed.}
\item{step}{Numeric; combined arithmetically in the body.}
\item{n_imp}{Coerced to integer by the body, with \code{as.integer}.}
\item{error_rate}{See Usage.}
\item{seed}{See Usage.}
}
\description{
Part of the mqtmpl_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{peak_lod}, \code{peak_position}, \code{position}, \code{lod}, \code{method_used}, \code{n_imputations}, \code{note}, \code{method}.
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
