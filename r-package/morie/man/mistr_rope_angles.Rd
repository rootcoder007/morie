% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mistr_rope_angles}
\alias{mistr_rope_angles}
\title{mistr_rope_angles}
\usage{
mistr_rope_angles(d, base = 10000) 
}
\arguments{
\item{d}{Numeric; combined arithmetically in the body.}
\item{base}{Numeric; combined arithmetically in the body. Defaults to \code{10000}.}
}
\description{
Part of the mistr_native implementation; see the file header for the source it follows.
}
\value{
A numeric value.
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
th <- mistr_rope_angles(8)
c(length(th), th[1] == 1)
}
