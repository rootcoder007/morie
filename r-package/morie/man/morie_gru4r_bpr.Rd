% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gru4r_bpr}
\alias{morie_gru4r_bpr}
\title{BPR ranking loss}
\usage{
morie_gru4r_bpr(r_target, r_negatives) 
}
\arguments{
\item{r_target}{Score of the target item.}
\item{r_negatives}{Numeric vector of negative-item scores.}
}
\description{
\eqn{-1/N_S \\sum_j \\log\\sigma(r_i - r_j)}, with the log floored at 1e-12 for numerical stability.
}
\value{
Scalar loss.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
morie_gru4r_bpr(V, V)
}
