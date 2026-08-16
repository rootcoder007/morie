% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mqtmpl_imputation_weights}
\alias{mqtmpl_imputation_weights}
\title{mqtmpl_imputation_weights}
\usage{
mqtmpl_imputation_weights(y, genotype_column, model_dimension = 2)
}
\arguments{
\item{y}{A vector; its length is taken.}
\item{genotype_column}{A vector; its length is taken.}
\item{model_dimension}{Numeric; combined arithmetically in the body. Defaults to \code{2}.}
}
\description{
Part of the mqtmpl_native implementation; see the file header for the source it follows.
}
\value{
A numeric value.
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
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
mqtmpl_imputation_weights(V, V)
}
