% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.hrzb1_score}
\alias{.hrzb1_score}
\title{Internal: Manski maximum-score objective (negative average}
\usage{
.hrzb1_score(b, ys, X)
}
\arguments{
\item{b}{A matrix; passed to \code{\%*\%}.}
\item{ys}{Numeric; combined arithmetically in the body.}
\item{X}{A matrix; passed to \code{\%*\%}.}
}
\description{
concordance of the sign of the index with the +-1 response). Extracted from the hrzb1() optimiser closures so it can be unit-tested directly.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
.hrzb1_score(b = c(1, 2, 3, 4, 5, 6, 7, 8), ys = c(1, 2, 3, 4, 5, 6, 7, 8), X = c(1, 2, 3, 4, 5, 6, 7, 8))
}
