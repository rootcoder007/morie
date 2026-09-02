% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{motfsr_log_odds_matrix}
\alias{motfsr_log_odds_matrix}
\title{motfsr_log_odds_matrix}
\usage{
motfsr_log_odds_matrix(motif, background) 
}
\arguments{
\item{motif}{Iterated over elementwise, with \code{lapply}.}
\item{background}{A vector; indexed elementwise.}
}
\description{
Part of the motfsr_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\references{
  Bailey & Elkan (1994) ISMB-94, 28-36.
  Bailey & Elkan (1995) ISMB-95, 21-29.
Base R only.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
motfsr_log_odds_matrix(V, V)
}
