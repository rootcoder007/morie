% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mpfn_sig}
\alias{mpfn_sig}
\title{mpfn_sig}
\usage{
mpfn_sig(x) 
}
\arguments{
\item{x}{Numeric; combined arithmetically in the body.}
}
\description{
Part of the mpfn_native implementation; see the file header for the source it follows.
}
\value{
One of two values, depending on the branch taken.
}
\references{
  Gilmer et al. (2017) "Neural Message Passing for Quantum Chemistry" arXiv:1704.01212
  Li et al. (2016) "Gated Graph Sequence Neural Networks" arXiv:1511.05493
  Vinyals et al. (2016) "Order Matters: Sequence to sequence for sets" arXiv:1511.06391
Base R only.
}
\keyword{internal}
\examples{
mpfn_sig(x = 5L)
}
