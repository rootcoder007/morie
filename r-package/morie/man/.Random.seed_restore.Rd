% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.Random.seed_restore}
\alias{.Random.seed_restore}
\title{.Random.seed_restore}
\usage{
.Random.seed_restore(old)
}
\arguments{
\item{old}{See Usage.}
}
\description{
Part of the xavir implementation; see the file header for the source it follows.
}
\value{
One of two values, depending on the branch taken.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.Random.seed_restore(V)
}
