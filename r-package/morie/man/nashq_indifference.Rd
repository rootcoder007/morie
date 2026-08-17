% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{nashq_indifference}
\alias{nashq_indifference}
\title{nashq_indifference}
\usage{
nashq_indifference(payoff, k)
}
\arguments{
\item{payoff}{A matrix; indexed by row and column.}
\item{k}{A count; the body uses it as \code{seq_len(...)}.}
}
\description{
Part of the nashq_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{[}.
}
\keyword{internal}
\examples{
A <- matrix(c(2, 0, 0, 1), 2, 2, byrow = TRUE)
nashq_indifference(A, 2)
}
