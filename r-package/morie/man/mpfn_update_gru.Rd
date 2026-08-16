% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mpfn_update_gru}
\alias{mpfn_update_gru}
\title{mpfn_update_gru}
\usage{
mpfn_update_gru(h, m, Wz, Uz, Wr, Ur, Wh, Uh)
}
\arguments{
\item{h}{A vector; its length is taken.}
\item{m}{Passed to \code{lin}.}
\item{Wz}{Passed to \code{lin}.}
\item{Uz}{Passed to \code{lin}.}
\item{Wr}{Passed to \code{lin}.}
\item{Ur}{Passed to \code{lin}.}
\item{Wh}{Passed to \code{lin}.}
\item{Uh}{Passed to \code{lin}.}
}
\description{
Part of the mpfn_native implementation; see the file header for the source it follows.
}
\value{
A numeric value.
}
\references{
  Gilmer et al. (2017) "Neural Message Passing for Quantum Chemistry" arXiv:1704.01212
  Li et al. (2016) "Gated Graph Sequence Neural Networks" arXiv:1511.05493
  Vinyals et al. (2016) "Order Matters: Sequence to sequence for sets" arXiv:1511.06391
Base R only.
}
\keyword{internal}
