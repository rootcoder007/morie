% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mpfn_readout}
\alias{mpfn_readout}
\title{mpfn_readout}
\usage{
mpfn_readout(H, how = "sum", H0 = NULL, i_fn = NULL, j_fn = NULL)
}
\arguments{
\item{H}{Iterated over elementwise, with \code{lapply}.}
\item{how}{One of \code{"gated"}, \code{"mean"}, \code{"sum"}. Defaults to \code{"sum"}.}
\item{H0}{Optional; may be \code{NULL}. A vector; indexed elementwise.}
\item{i_fn}{Defaults to \code{NULL}.}
\item{j_fn}{Defaults to \code{NULL}.}
}
\description{
Part of the mpfn_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{acc}, as built in the body.
}
\references{
  Gilmer et al. (2017) "Neural Message Passing for Quantum Chemistry" arXiv:1704.01212
  Li et al. (2016) "Gated Graph Sequence Neural Networks" arXiv:1511.05493
  Vinyals et al. (2016) "Order Matters: Sequence to sequence for sets" arXiv:1511.06391
Base R only.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
mpfn_readout(V)
}
