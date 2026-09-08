% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_tqlld_lloyd_max_codebook}
\alias{morie_tqlld_lloyd_max_codebook}
\alias{morie_tqlld_tqlld}
\alias{morie_tqlld_turboquant_lloyd_max_codebook}
\title{Lloyd-Max optimal scalar codebook}
\usage{
morie_tqlld_lloyd_max_codebook(levels = 4L, source = "gaussian", data = NULL, lo = NULL, hi = NULL, 
    max_iter = 200L, tol = 1e-12, n_grid = 20000L) 

morie_tqlld_tqlld(levels = 4L, source = "gaussian", data = NULL, lo = NULL, hi = NULL, 
    max_iter = 200L, tol = 1e-12, n_grid = 20000L) 

morie_tqlld_turboquant_lloyd_max_codebook(levels = 4L, source = "gaussian", data = NULL, lo = NULL, hi = NULL, 
    max_iter = 200L, tol = 1e-12, n_grid = 20000L) 
}
\arguments{
\item{levels}{Number of codewords.}
\item{source}{One of \code{"gaussian"}, \code{"empirical"},   \code{"uniform"}.}
\item{data}{Empirical samples for \code{source = "empirical"}.}
\item{lo,hi}{Range for the uniform source.}
\item{max_iter}{Maximum alternations.}
\item{tol}{Convergence tolerance on codeword shift and distortion   change.}
\item{n_grid}{Number of quadrature points for the Gaussian source.}
}
\description{
Returns a list with \code{codebook}, \code{boundaries}, \code{distortion}, \code{distortion_history}, \code{iterations}, \code{converged}, \code{source}, \code{levels} and \code{method}. The uniform source has a closed form; the Gaussian and empirical sources iterate the nearest-neighbour and centroid conditions until both hold.
}
\value{
A list with \code{estimate}, \code{codebook}, \code{boundaries}, \code{distortion}, \code{distortion_history}, \code{iterations}, \code{converged}, \code{source}, \code{levels}, \code{method}.
}
\keyword{internal}
\examples{
morie_tqlld_lloyd_max_codebook()
}
