% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{noisy_argmax}
\alias{noisy_argmax}
\title{Equation 1: argmax_j {n_j + Lap(1/gamma)}. One independent}
\usage{
noisy_argmax(counts, gamma, seed = 0)
}
\arguments{
\item{counts}{See Usage.}
\item{gamma}{See Usage.}
\item{seed}{Defaults to \code{0}.}
}
\description{
Laplace draw per class with scale 1/gamma.
}
\value{
A numeric value.
}
\keyword{internal}
