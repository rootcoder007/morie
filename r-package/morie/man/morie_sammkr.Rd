% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_sammkr}
\alias{morie_sammkr}
\title{morie_sammkr}
\usage{
morie_sammkr(masks, predicted_iou, target = NULL) 
}
\arguments{
\item{masks}{Passed to \code{rank_masks}.}
\item{predicted_iou}{Passed to \code{rank_masks}.}
\item{target}{Passed to \code{rank_masks}.}
}
\description{
Part of the sammkr_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{rank_masks}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
morie_sammkr(V, V)
}
