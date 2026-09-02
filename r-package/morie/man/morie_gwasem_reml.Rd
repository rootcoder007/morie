% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gwasem_reml}
\alias{morie_gwasem_reml}
\title{REML estimation of the variance components}
\usage{
morie_gwasem_reml(y, Kinship, covariates = NULL, ml = FALSE) 
}
\arguments{
\item{y}{Phenotype vector.}
\item{Kinship}{Relatedness matrix.}
\item{covariates}{Optional covariate matrix.}
\item{ml}{Use maximum likelihood.}
}
\description{
Step 2: estimate \eqn{\sigma_a^2, \sigma_e^2} on the spectral basis of \eqn{\hat S_N}, plus a null comparison.
}
\value{
List with \code{sigma_a2}, \code{sigma_e2}, \code{delta},   \code{pseudo_heritability}, \code{loglik}, \code{loglik_null},   \code{lrt}, \code{evals}, \code{evecs}, \code{kinship_normalized}   and \code{shift}.
}
\keyword{internal}
