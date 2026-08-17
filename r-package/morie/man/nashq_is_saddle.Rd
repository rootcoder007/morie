% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{nashq_is_saddle}
\alias{nashq_is_saddle}
\title{nashq_is_saddle}
\usage{
nashq_is_saddle(A, B, p, q, tol)
}
\arguments{
\item{A}{A matrix; passed to \code{nrow}.}
\item{B}{Passed to \code{nashq_payoff}.}
\item{p}{Passed to \code{nashq_payoff}.}
\item{q}{Passed to \code{nashq_payoff}.}
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
A <- matrix(c(1, -1, -1, 1), 2, 2, byrow = TRUE)
B <- -A
nashq_is_saddle(A, B, c(0.5, 0.5), c(0.5, 0.5), 1e-9)
}
