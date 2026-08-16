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
\item{H}{See Usage.}
\item{adj}{See Usage.}
\item{edge_features}{See Usage.}
\item{perm}{See Usage.}
\item{T}{Defaults to \code{3}.}
\item{how}{Defaults to \code{"sum"}.}
\item{tol}{Defaults to \code{1e-09}.}
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
