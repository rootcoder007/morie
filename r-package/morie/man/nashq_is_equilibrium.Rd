% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{nashq_is_equilibrium}
\alias{nashq_is_equilibrium}
\title{nashq_is_equilibrium}
\usage{
nashq_is_equilibrium(A, B, p, q, tol) 
}
\arguments{
\item{A}{A matrix; indexed by row and column.}
\item{B}{A matrix; indexed by row and column.}
\item{p}{Numeric; combined arithmetically in the body.}
\item{q}{Numeric; combined arithmetically in the body.}
\item{tol}{Numeric; combined arithmetically in the body.}
}
\description{
Part of the nashq_native implementation; see the file header for the source it follows.
}
\value{
A logical value.
}
\keyword{internal}
\examples{
A <- matrix(c(2, 0, 0, 1), 2, 2, byrow = TRUE)
B <- matrix(c(1, 0, 0, 2), 2, 2, byrow = TRUE)
nashq_is_equilibrium(A, B, c(1, 0), c(1, 0), 1e-9)
}
