% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_rank_normalize}
\alias{.morie_rank_normalize}
\title{.morie_rank_normalize}
\usage{
.morie_rank_normalize(C)
}
\arguments{
\item{C}{A matrix; passed to \code{nrow}.}
}
\description{
Part of the mcmc_native implementation; see the file header for the source it follows.
}
\value{
A matrix, from \code{matrix}.
}
\keyword{internal}
\examples{
M <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
.morie_rank_normalize(M)
}
