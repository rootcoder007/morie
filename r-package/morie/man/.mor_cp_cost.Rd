% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.mor_cp_cost}
\alias{.mor_cp_cost}
\title{Segment cost of the 0-based half-open block [a, b)}
\usage{
.mor_cp_cost(tab, cost)
}
\arguments{
\item{tab}{A list; the body reads \code{$cs}, \code{$css} from it.}
\item{cost}{One of \code{"mean"}, \code{"meanvar"}.}
}
\description{
Part of the pelt_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{function}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.mor_cp_cost(V, V)
}
