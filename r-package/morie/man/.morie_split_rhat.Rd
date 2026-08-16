% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_split_rhat}
\alias{.morie_split_rhat}
\title{.morie_split_rhat}
\usage{
.morie_split_rhat(chains, rank_normalized = TRUE)
}
\arguments{
\item{chains}{A count; the body uses it as \code{matrix(...)}.}
\item{rank_normalized}{A flag; the body branches on it. Defaults to \code{TRUE}.}
}
\description{
Part of the mcmc_native implementation; see the file header for the source it follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.morie_split_rhat(V)
}
