% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.as_item_matrix}
\alias{.as_item_matrix}
\title{.as_item_matrix}
\usage{
.as_item_matrix(data)
}
\arguments{
\item{data}{A matrix; passed to \code{as.matrix}.}
}
\description{
Part of the psymet implementation; see the file header for the source it follows.
}
\value{
The value of \code{X}, as built in the body.
}
\references{
  Cronbach (1951). Psychometrika, 16(3), 297-334.
  McDonald (1999). Test Theory: A Unified Treatment.
  Revelle (2024). psych R package.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.as_item_matrix(V)
}
