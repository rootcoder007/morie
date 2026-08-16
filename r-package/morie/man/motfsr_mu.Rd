% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{motfsr_mu}
\alias{motfsr_mu}
\title{motfsr_mu}
\usage{
motfsr_mu(coded, L)
}
\arguments{
\item{coded}{See Usage.}
\item{L}{A count; the body uses it as \code{numeric(...)}.}
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
motfsr_mu(coded = c(1, 2, 3, 4, 5, 6, 7, 8), L = 5L)
}
