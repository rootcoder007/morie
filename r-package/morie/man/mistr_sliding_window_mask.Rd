% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mistr_sliding_window_mask}
\alias{mistr_sliding_window_mask}
\title{mistr_sliding_window_mask}
\usage{
mistr_sliding_window_mask(L, window, causal = TRUE)
}
\arguments{
\item{L}{A count; the body uses it as \code{seq_len(...)}.}
\item{window}{Passed to \code{<}.}
\item{causal}{A flag; the body branches on it. Defaults to \code{TRUE}.}
}
\description{
Part of the mistr_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{mask}, as built in the body.
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
mistr_sliding_window_mask(L = c(1, 2, 3, 4, 5, 6, 7, 8), window = 5L)
}
