% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{momento_mask_patches}
\alias{momento_mask_patches}
\title{momento_mask_patches}
\usage{
momento_mask_patches(patches, mask_idx, fill = 0)
}
\arguments{
\item{patches}{A vector; its length is taken and its elements indexed.}
\item{mask_idx}{See Usage.}
\item{fill}{A count; the body uses it as \code{rep(...)}. Defaults to \code{0}.}
}
\description{
Part of the momento_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{masked}, \code{mask}, \code{mask_idx}, \code{mask_rate}, \code{n_patches}.
}
\references{
  Goswami et al. (2024) "MOMENT" arXiv:2402.03885
  Devlin et al. (2019) "BERT" arXiv:1810.04805
  Nie et al. (2023) "PatchTST" arXiv:2211.14730
Base R only.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
M <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
momento_mask_patches(V, M)
}
