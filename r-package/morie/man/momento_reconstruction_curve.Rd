% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{momento_reconstruction_curve}
\alias{momento_reconstruction_curve}
\title{momento_reconstruction_curve}
\usage{
momento_reconstruction_curve(patches, reconstructor, rates, seed = 0) 
}
\arguments{
\item{patches}{Iterated over elementwise, with \code{lapply}.}
\item{reconstructor}{Accepted by the signature and not used anywhere in the body.}
\item{rates}{See Usage.}
\item{seed}{Passed to \code{set.seed}. Defaults to \code{0}.}
}
\description{
Part of the momento_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{curve}, \code{n_patches}, \code{rates}, \code{mse}.
}
\references{
  Goswami et al. (2024) "MOMENT" arXiv:2402.03885
  Devlin et al. (2019) "BERT" arXiv:1810.04805
  Nie et al. (2023) "PatchTST" arXiv:2211.14730
Base R only.
}
\keyword{internal}
