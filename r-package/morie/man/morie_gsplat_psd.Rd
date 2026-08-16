% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gsplat_psd}
\alias{morie_gsplat_psd}
\title{Positive semi-definiteness check}
\usage{
morie_gsplat_psd(S, tol = -1e-09)
}
\arguments{
\item{S}{3x3 symmetric matrix.}
\item{tol}{Tolerance for \code{min_eigenvalue >= tol}.}
}
\description{
Computes the eigenvalues of a symmetric matrix and tests the smallest against a tolerance.
}
\value{
A list with \code{eigenvalues}, \code{min_eigenvalue} and   \code{psd}.
}
\keyword{internal}
