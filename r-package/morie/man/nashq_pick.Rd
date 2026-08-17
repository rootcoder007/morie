% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{nashq_pick}
\alias{nashq_pick}
\title{nashq_pick}
\usage{
nashq_pick(M, A, who, epsilon, rng)
}
\arguments{
\item{M}{Passed to \code{rowMeans}.}
\item{A}{A vector; its length is taken.}
\item{who}{Passed to \code{==}.}
\item{epsilon}{Passed to \code{<}.}
\item{rng}{Accepted by the signature and not used anywhere in the body.}
}
\description{
Part of the nashq_native implementation; see the file header for the source it follows.
}
\value{
One of two values, depending on the branch taken.
}
\keyword{internal}
\examples{
A <- matrix(c(2, 0, 0, 1), 2, 2, byrow = TRUE)
nashq_pick(A, 0:1, 0L, 0.0, function() 0.99)
}
