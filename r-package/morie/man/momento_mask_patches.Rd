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
\item{patches}{See Usage.}
\item{mask_idx}{See Usage.}
\item{fill}{Defaults to \code{0}.}
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
