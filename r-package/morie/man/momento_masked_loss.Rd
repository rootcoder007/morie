% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{momento_masked_loss}
\alias{momento_masked_loss}
\title{momento_masked_loss}
\usage{
momento_masked_loss(truth, reconstruction, mask) 
}
\arguments{
\item{truth}{A vector; its length is taken and its elements indexed.}
\item{reconstruction}{A vector; its length is taken and its elements indexed.}
\item{mask}{A vector; its length is taken and its elements indexed.}
}
\description{
Part of the momento_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{mse}, \code{n_scored}, \code{scored}.
}
\references{
  Goswami et al. (2024) "MOMENT" arXiv:2402.03885
  Devlin et al. (2019) "BERT" arXiv:1810.04805
  Nie et al. (2023) "PatchTST" arXiv:2211.14730
Base R only.
}
\keyword{internal}
\examples{
momento_masked_loss(truth = c(1, 2, 3, 4, 5, 6, 7, 8),
  reconstruction = c(1, 2, 3, 4, 5, 6, 7, 8), mask = c(1, 2, 3, 4, 5, 6, 7, 8))
}
