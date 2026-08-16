% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{qini_coefficient}
\alias{qini_coefficient}
\title{Qini coefficient (RATE with alpha(u) = u)}
\usage{
qini_coefficient(scores, priority)
}
\description{
Qini coefficient (RATE with alpha(u) = u)
}
\value{
The value of \code{$}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
qini_coefficient(V, V)
}
