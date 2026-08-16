% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mqtmpl_scan_cim}
\alias{mqtmpl_scan_cim}
\title{mqtmpl_scan_cim}
\usage{
mqtmpl_scan_cim(y, markers, positions, cofactors = list(), window = 0, step = 0.02)
}
\arguments{
\item{y}{See Usage.}
\item{markers}{See Usage.}
\item{positions}{See Usage.}
\item{cofactors}{Defaults to \code{list()}.}
\item{window}{Defaults to \code{0}.}
\item{step}{Defaults to \code{0.02}.}
}
\description{
Part of the mqtmpl_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{peak_lod}, \code{peak_position}, \code{position}, \code{lod}, \code{fit}.
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
