% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gtrf_sign_flip}
\alias{morie_gtrf_sign_flip}
\title{Random sign flip}
\usage{
morie_gtrf_sign_flip(pe, rng)
}
\arguments{
\item{pe}{Positional encoding matrix (n x d).}
\item{rng}{Generator environment (shared with the Python arm).}
}
\description{
Flips each eigenvector\'s sign with probability 1/2, per the paper.
}
\value{
Sign-flipped encoding.
}
\keyword{internal}
