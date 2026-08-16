% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gwasem_kinship_ibs}
\alias{morie_gwasem_kinship_ibs}
\title{IBS relatedness matrix}
\usage{
morie_gwasem_kinship_ibs(genotypes)
}
\arguments{
\item{genotypes}{n x m minor-allele-count matrix.}
}
\description{
\eqn{\hat S_{ik} = 1 - \frac{1}{2M}\sum_j |g_{ij} - g_{kj}|}.
}
\value{
n x n symmetric numeric matrix.
}
\keyword{internal}
\examples{
M <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
morie_gwasem_kinship_ibs(M)
}
