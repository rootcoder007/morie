% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{motfsr_bayes_threshold}
\alias{motfsr_bayes_threshold}
\title{motfsr_bayes_threshold}
\usage{
motfsr_bayes_threshold(lambda1, loss = NULL)
}
\arguments{
\item{lambda1}{Numeric; combined arithmetically in the body.}
\item{loss}{Optional; may be \code{NULL}. A vector; indexed elementwise.}
}
\description{
Part of the motfsr_native implementation; see the file header for the source it follows.
}
\value{
A numeric value.
}
\references{
  Bailey & Elkan (1994) ISMB-94, 28-36.
  Bailey & Elkan (1995) ISMB-95, 21-29.
Base R only.
}
\keyword{internal}
\examples{
set.seed(1)
r <- motfsr_bayes_threshold(lambda1 = 0.5); TRUE
}
