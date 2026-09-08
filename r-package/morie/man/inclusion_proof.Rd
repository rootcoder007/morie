% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{inclusion_proof}
\alias{inclusion_proof}
\title{Build an inclusion proof (audit path) for the given leaf}
\usage{
inclusion_proof(leaves, index) 
}
\description{
Build an inclusion proof (audit path) for the given leaf
}
\arguments{
\item{leaves}{Argument \code{leaves}; see Usage.}
\item{index}{Argument \code{index}; see Usage.}
}
\value{
A list with \code{path}, \code{path_hex}, \code{length}, \code{index}, \code{size}, \code{note}.
}
\keyword{internal}
\examples{
if (morie_crypto_sodium_available()) {
  leaves <- list("a", "b", "c", "d", "e")
  pr <- inclusion_proof(leaves, index = 2)
  c(pr$index, pr$size, pr$length)
}
}
