% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{decay_diagnostics}
\alias{decay_diagnostics}
\title{Realised tail vs the expected geometric tail}
\usage{
decay_diagnostics(weights, alpha) 
}
\description{
Realised tail vs the expected geometric tail
}
\arguments{
\item{weights}{Argument \code{weights}; see Usage.}
\item{alpha}{Argument \code{alpha}; see Usage.}
}
\value{
A list with \code{realised_tail}, \code{expected_tail}, \code{ratio}, \code{largest_index}, \code{monotone}, \code{note}.
}
\keyword{internal}
\examples{
decay_diagnostics(weights = c(1, 2, 3, 4, 5, 6, 7, 8), alpha = 0.5)
}
