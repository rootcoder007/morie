% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{nashq_payoff}
\alias{nashq_payoff}
\title{nashq_payoff}
\usage{
nashq_payoff(M, p, q) 
}
\arguments{
\item{M}{A matrix; indexed by row and column.}
\item{p}{A vector; its length is taken and its elements indexed.}
\item{q}{A vector; its length is taken and its elements indexed.}
}
\description{
Part of the nashq_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{tot}, as built in the body.
}
\keyword{internal}
\examples{
A <- matrix(c(2, 0, 0, 1), 2, 2, byrow = TRUE)
nashq_payoff(A, c(0.5, 0.5), c(0.5, 0.5))
}
