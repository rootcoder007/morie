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
\item{h_v}{See Usage.}
\item{h_w}{See Usage.}
\item{e_vw}{See Usage.}
\item{A}{Defaults to \code{NULL}.}
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
