% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{moments_accountant}
\alias{moments_accountant}
\title{Compose the per-query moment bounds and convert to (eps, delta)}
\usage{
moments_accountant(vote_counts, gamma, delta, lambdas = NULL, data_dependent = TRUE) 
}
\arguments{
\item{vote_counts}{A vector; its length is taken.}
\item{gamma}{Numeric; combined arithmetically in the body.}
\item{delta}{Numeric; combined arithmetically in the body.}
\item{lambdas}{Optional; may be \code{NULL}. Coerced to integer by the body, with \code{as.integer}.}
\item{data_dependent}{A flag; the body branches on it. Defaults to \code{TRUE}.}
}
\description{
Part of the pate_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{epsilon}, \code{lambda}, \code{alpha}, \code{delta}, \code{queries}, \code{used}.
}
\keyword{internal}
\examples{
moments_accountant(vote_counts = c(1, 2, 3, 4, 5, 6, 7, 8), gamma = 0.5, delta = 0.5)
}
