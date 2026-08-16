% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gwasem}
\alias{morie_gwasem}
\title{EMMAX genome-wide association scan}
\usage{
morie_gwasem(y, genotypes, kinship = NULL, covariates = NULL, trait = "quantitative", test = "f", ml = FALSE, per_marker_reml = FALSE, min_maf = 0)
}
\arguments{
\item{y}{Phenotype vector.}
\item{genotypes}{Individual x marker matrix (minor allele counts).}
\item{kinship}{Optional n x n relatedness matrix; computed by IBS   when omitted.}
\item{covariates}{Optional covariate matrix.}
\item{trait}{"quantitative" or "binary".}
\item{test}{"f" (GLS F-test) or "score".}
\item{ml}{Use maximum likelihood instead of REML.}
\item{per_marker_reml}{Re-estimate variance components per marker   (the exact EMMA model, off by default).}
\item{min_maf}{Skip markers below this minor allele frequency.}
}
\description{
The three-step procedure: (1) Gower-normalise a relatedness matrix \eqn{\hat S}, (2) estimate \eqn{\sigma_a^2, \sigma_e^2} once by REML/ML in \eqn{\mathrm{Var}(Y) = \sigma_a^2 \hat S_N + \sigma_e^2 I}, and (3) GLS F-test (or score test) at every marker with that fixed V. Step 2 happening once is the eXpedited part; the expensive per-marker REML is available as an exact alternative.
}
\value{
A list with \code{beta}, \code{se}, \code{stat},   \code{pvalue}, \code{variance_components},   \code{pseudo_heritability}, \code{lambda_gc}, \code{skipped},   \code{n}, \code{n_markers}, \code{test}, \code{trait},   \code{per_marker_reml}, \code{note} and \code{method}.
}
\keyword{internal}
