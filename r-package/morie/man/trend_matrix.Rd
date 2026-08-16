% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{trend_matrix}
\alias{trend_matrix}
\title{Design columns for k, m and each delta_j}
\usage{
trend_matrix(t, cps)
}
\arguments{
\item{t}{Numeric vector of times.}
\item{cps}{Numeric vector of changepoints.}
}
\description{
The delta column is \code{a_j(t)(t - s_j)}, which already carries the \code{-s_j delta_j} offset -- so continuity holds by construction.
}
\value{
A list of rows, each a numeric vector.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
trend_matrix(V, V)
}
