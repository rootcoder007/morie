% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mistr_apply_rope}
\alias{mistr_apply_rope}
\title{mistr_apply_rope}
\usage{
mistr_apply_rope(x, pos, theta = NULL, base = 10000)
}
\arguments{
\item{x}{See Usage.}
\item{pos}{See Usage.}
\item{theta}{Defaults to \code{NULL}.}
\item{base}{Defaults to \code{10000}.}
}
\description{
Part of the mistr_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
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
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
mistr_apply_rope(V, V)
}
