% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_tqlld_quantize_with_codebook}
\alias{morie_tqlld_quantize_with_codebook}
\title{Quantise samples with a fixed codebook}
\usage{
morie_tqlld_quantize_with_codebook(x, codebook) 
}
\description{
Quantise samples with a fixed codebook
}
\arguments{
\item{x}{Argument \code{x}; see Usage.}
\item{codebook}{Argument \code{codebook}; see Usage.}
}
\value{
A list with \code{estimate}, \code{indices}, \code{values},   \code{mse}, \code{levels}, \code{method}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
morie_tqlld_quantize_with_codebook(V, V)
}
