% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{singular_value_adjust}
\alias{singular_value_adjust}
\title{SVD cap on the prewhitening matrix}
\usage{
singular_value_adjust(a, cap = .EIGENVALUE_CAP) 
}
\description{
SVD cap on the prewhitening matrix
}
\arguments{
\item{a}{Argument \code{a}; see Usage.}
\item{cap}{Argument \code{cap}; see Usage.}
}
\value{
The value of \code{\%*\%}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
singular_value_adjust(V)
}
