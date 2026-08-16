% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.psych_or_stop}
\alias{.psych_or_stop}
\title{.psych_or_stop}
\usage{
.psych_or_stop(fn)
}
\arguments{
\item{fn}{Passed to \code{sprintf}.}
}
\description{
Part of the psymet implementation; see the file header for the source it follows.
}
\value{
One of two values, depending on the branch taken.
}
\references{
  Cronbach (1951). Psychometrika, 16(3), 297-334.
  McDonald (1999). Test Theory: A Unified Treatment.
  Revelle (2024). psych R package.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.psych_or_stop(V)
}
