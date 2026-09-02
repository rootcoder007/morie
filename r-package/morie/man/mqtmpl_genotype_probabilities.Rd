% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mqtmpl_genotype_probabilities}
\alias{mqtmpl_genotype_probabilities}
\title{Qk left, qk+1 right}
\usage{
mqtmpl_genotype_probabilities(s_left, s_right, r_left, r_right) 
}
\arguments{
\item{s_left}{Accepted by the signature and not used anywhere in the body.}
\item{s_right}{Accepted by the signature and not used anywhere in the body.}
\item{r_left}{Numeric; combined arithmetically in the body.}
\item{r_right}{Numeric; combined arithmetically in the body.}
}
\description{
Part of the mqtmpl_native implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{c}.
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
\examples{
mqtmpl_genotype_probabilities(s_left = 0, s_right = 1,
                             r_left = 0.1, r_right = 0.1)
}
