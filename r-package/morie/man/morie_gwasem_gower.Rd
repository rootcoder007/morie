% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gwasem_gower}
\alias{morie_gwasem_gower}
\title{Gower normalisation of a relatedness matrix}
\usage{
morie_gwasem_gower(S) 
}
\arguments{
\item{S}{n x n relatedness matrix.}
}
\description{
Equation 5: \eqn{\hat S_N = (n-1)\hat S / \mathrm{Tr}(P\hat S P)} with \eqn{P = I - \mathbf{1}\mathbf{1}\'/n}, so \eqn{\sigma_a^2} is on the scale of the phenotypic variance.
}
\value{
Normalised matrix.
}
\keyword{internal}
\examples{
M <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
morie_gwasem_gower(M)
}
