% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mpfn_message}
\alias{mpfn_message}
\title{mpfn_message}
\usage{
mpfn_message(h_v, h_w, e_vw, A = NULL) 
}
\arguments{
\item{h_v}{Accepted by the signature and not used anywhere in the body.}
\item{h_w}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{e_vw}{A vector; its length is taken and its elements indexed.}
\item{A}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
}
\description{
Part of the mpfn_native implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{as.numeric}.
}
\references{
  Gilmer et al. (2017) "Neural Message Passing for Quantum Chemistry" arXiv:1704.01212
  Li et al. (2016) "Gated Graph Sequence Neural Networks" arXiv:1511.05493
  Vinyals et al. (2016) "Order Matters: Sequence to sequence for sets" arXiv:1511.06391
Base R only.
}
\keyword{internal}
\examples{
mpfn_message(h_v = c(1, 2, 3, 4, 5, 6, 7, 8), h_w = c(1, 2, 3, 4, 5, 6, 7, 8),
  e_vw = c(1, 2, 3, 4, 5, 6, 7, 8))
}
