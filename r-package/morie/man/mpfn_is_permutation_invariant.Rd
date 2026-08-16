% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mpfn_is_permutation_invariant}
\alias{mpfn_is_permutation_invariant}
\title{mpfn_is_permutation_invariant}
\usage{
mpfn_is_permutation_invariant(H, adj, edge_features, perm, T = 3, how = "sum", tol = 1e-09)
}
\arguments{
\item{H}{A vector; its length is taken and its elements indexed.}
\item{adj}{A vector; its length is taken and its elements indexed.}
\item{edge_features}{A vector; its length is taken and its elements indexed.}
\item{perm}{A vector; indexed elementwise.}
\item{T}{Passed to \code{mpfn_message_passing}. Defaults to \code{3}.}
\item{how}{Passed to \code{mpfn_readout}. Defaults to \code{"sum"}.}
\item{tol}{Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{1e-09}.}
}
\description{
Part of the mpfn_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{invariant}, \code{max_deviation}, \code{readout}.
}
\references{
  Gilmer et al. (2017) "Neural Message Passing for Quantum Chemistry" arXiv:1704.01212
  Li et al. (2016) "Gated Graph Sequence Neural Networks" arXiv:1511.05493
  Vinyals et al. (2016) "Order Matters: Sequence to sequence for sets" arXiv:1511.06391
Base R only.
}
\keyword{internal}
\examples{
mpfn_is_permutation_invariant(H = 0.5, adj = data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9)), edge_features = data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9)), perm = c(1, 2, 3, 4, 5, 6, 7, 8))
}
