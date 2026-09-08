% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{nashq_solve}
\alias{nashq_solve}
\title{nashq_solve}
\usage{
nashq_solve(A, b) 
}
\arguments{
\item{A}{Passed to \code{cbind}.}
\item{b}{A vector; its length is taken.}
}
\description{
Part of the nashq_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{[}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
nashq_solve(V, V)
}
