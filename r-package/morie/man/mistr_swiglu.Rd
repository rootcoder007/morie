% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mistr_swiglu}
\alias{mistr_swiglu}
\title{mistr_swiglu}
\usage{
mistr_swiglu(x, W1, W2, W3)
}
\arguments{
\item{x}{A matrix; passed to \code{crossprod}.}
\item{W1}{A matrix; passed to \code{crossprod}.}
\item{W2}{A matrix; passed to \code{crossprod}.}
\item{W3}{A matrix; passed to \code{crossprod}.}
}
\description{
Part of the mistr_native implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{as.numeric}.
}
\references{
  Jiang et al. (2023) "Mistral 7B" arXiv:2310.06825
  Su et al. (2024) "RoFormer" Neurocomputing 568, 127063
  Shazeer (2020) "GLU Variants" arXiv:2002.05202
  Zhang & Sennrich (2019) "RMSNorm" arXiv:1910.07467
  Ainslie et al. (2023) "GQA" arXiv:2305.13245
  Beltagy et al. (2020) "Longformer" arXiv:2004.05150
Base R only.
}
\keyword{internal}
