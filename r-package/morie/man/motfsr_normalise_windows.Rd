% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{motfsr_normalise_windows}
\alias{motfsr_normalise_windows}
\title{motfsr_normalise_windows}
\usage{
motfsr_normalise_windows(z, w, max_sweeps = 100) 
}
\arguments{
\item{z}{A vector; its length is taken and its elements indexed.}
\item{w}{A count; the body uses it as \code{seq_len(...)}.}
\item{max_sweeps}{A count; the body uses it as \code{seq_len(...)}. Defaults to \code{100}.}
}
\description{
Part of the motfsr_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{z}, as built in the body.
}
\references{
  Bailey & Elkan (1994) ISMB-94, 28-36.
  Bailey & Elkan (1995) ISMB-95, 21-29.
Base R only.
}
\keyword{internal}
\examples{
motfsr_normalise_windows(z = c(1, 2, 3, 4, 5, 6, 7, 8), w = 5L)
}
