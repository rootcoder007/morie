% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{motfsr_uniform_theta}
\alias{motfsr_uniform_theta}
\title{motfsr_uniform_theta}
\usage{
motfsr_uniform_theta(w, L, mu)
}
\arguments{
\item{w}{See Usage.}
\item{L}{See Usage.}
\item{mu}{See Usage.}
}
\description{
Part of the motfsr_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{theta}, as built in the body.
}
\references{
  Bailey & Elkan (1994) ISMB-94, 28-36.
  Bailey & Elkan (1995) ISMB-95, 21-29.
Base R only.
}
\keyword{internal}
\examples{
motfsr_uniform_theta(w = 5L, L = c(1, 2, 3, 4, 5, 6, 7, 8), mu = c(1, 2, 3, 4, 5, 6, 7, 8))
}
