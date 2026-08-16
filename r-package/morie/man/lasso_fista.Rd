% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{lasso_fista}
\alias{lasso_fista}
\title{Lasso via ISTA / FISTA}
\usage{
lasso_fista(A, b, lam, max_iter = 500L, tol = 1e-10, accelerate = TRUE)
}
\arguments{
\item{A}{Numeric matrix.}
\item{b}{Numeric vector.}
\item{lam}{Non-negative penalty.}
\item{max_iter}{Integer, maximum iterations.}
\item{tol}{Stop when the step is at most \code{tol}.}
\item{accelerate}{If \code{TRUE} run FISTA, otherwise ISTA.}
}
\description{
Minimises \code{1/2 ||Ax - b||^2 + lam * ||x||_1}; the prox is soft-thresholding at \code{lam * t}, not at \code{t}.
}
\value{
A list with \code{estimate}, \code{x}, \code{fun},   \code{objective}, \code{iterations}, \code{L}, \code{accelerated},   \code{converged}, \code{method}, \code{lambda}, \code{L}.
}
\keyword{internal}
