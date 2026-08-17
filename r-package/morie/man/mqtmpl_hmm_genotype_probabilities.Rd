% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mqtmpl_hmm_genotype_probabilities}
\alias{mqtmpl_hmm_genotype_probabilities}
\title{mqtmpl_hmm_genotype_probabilities}
\usage{
mqtmpl_hmm_genotype_probabilities(genotypes, positions, error_rate = 0)
}
\arguments{
\item{genotypes}{Iterated over elementwise, with \code{sapply}.}
\item{positions}{A vector; its length is taken and its elements indexed.}
\item{error_rate}{Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{0}.}
}
\description{
Part of the mqtmpl_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
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
set.seed(1)
pos <- c(0, 10, 20, 30, 40); m <- length(pos)
G <- matrix(0L, 8, m); G[, 1] <- rbinom(8, 1, 0.5)
for (j in 2:m) { r <- 0.5 * (1 - exp(-2 * (pos[j] - pos[j - 1]) / 100))
  fl <- rbinom(8, 1, r); G[, j] <- ifelse(fl == 1, 1L - G[, j - 1], G[, j - 1]) }
genos <- lapply(seq_len(8), function(i) G[i, ])
p <- mqtmpl_hmm_genotype_probabilities(genos, pos, error_rate = 0.01)
length(p) == 8L
}
