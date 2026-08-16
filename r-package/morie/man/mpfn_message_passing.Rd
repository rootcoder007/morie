% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mpfn_message_passing}
\alias{mpfn_message_passing}
\title{mpfn_message_passing}
\usage{
mpfn_message_passing(H0, adj, edge_features, T = 3, A = NULL, update = NULL)
}
\arguments{
\item{H0}{Iterated over elementwise, with \code{lapply}.}
\item{adj}{A vector; indexed elementwise.}
\item{edge_features}{A vector; indexed elementwise.}
\item{T}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{3}.}
\item{A}{Defaults to \code{NULL}.}
\item{update}{Defaults to \code{NULL}.}
}
\description{
Part of the mpfn_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{H}, as built in the body.
}
\references{
  Gilmer et al. (2017) "Neural Message Passing for Quantum Chemistry" arXiv:1704.01212
  Li et al. (2016) "Gated Graph Sequence Neural Networks" arXiv:1511.05493
  Vinyals et al. (2016) "Order Matters: Sequence to sequence for sets" arXiv:1511.06391
Base R only.
}
\keyword{internal}
\examples{
mpfn_message_passing(H0 = c(1, 2, 3, 4, 5, 6, 7, 8), adj = data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9)), edge_features = c(1, 2, 3, 4, 5, 6, 7, 8))
}
