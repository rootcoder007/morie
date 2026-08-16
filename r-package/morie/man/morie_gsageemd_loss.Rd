% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gsageemd_loss}
\alias{morie_gsageemd_loss}
\title{Unsupervised graph-based loss}
\usage{
morie_gsageemd_loss(z_u, z_v, z_negatives)
}
\arguments{
\item{z_u}{Anchor embedding.}
\item{z_v}{Positive (neighbour) embedding.}
\item{z_negatives}{List of negative embeddings.}
}
\description{
Sec. 3.2 of Hamilton et al. 2017: nearby nodes agree, sampled negatives disagree.
}
\value{
Scalar loss.
}
\keyword{internal}
\examples{
morie_gsageemd_loss(z_u = c(1, 2, 3, 4, 5, 6, 7, 8), z_v = c(1, 2, 3, 4, 5, 6, 7, 8), z_negatives = c(1, 2, 3, 4, 5, 6, 7, 8))
}
