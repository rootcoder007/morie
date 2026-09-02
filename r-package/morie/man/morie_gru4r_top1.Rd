% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gru4r_top1}
\alias{morie_gru4r_top1}
\title{TOP1 ranking loss}
\usage{
morie_gru4r_top1(r_target, r_negatives, regularize = TRUE) 
}
\arguments{
\item{r_target}{Score of the target item.}
\item{r_negatives}{Numeric vector of negative-item scores.}
\item{regularize}{Include the sigma(r_j^2) regulariser.}
}
\description{
The smoothed relative rank \eqn{\\sigma(r_j - r_i)} plus, when \code{regularize=TRUE}, the load-bearing \eqn{\\sigma(r_j^2)} term that stops scores from running away.
}
\value{
Scalar loss.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
morie_gru4r_top1(V, V)
}
