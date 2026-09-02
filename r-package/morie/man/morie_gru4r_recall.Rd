% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gru4r_recall}
\alias{morie_gru4r_recall}
\title{Recall at k}
\usage{
morie_gru4r_recall(ranked, target, kk = 20) 
}
\arguments{
\item{ranked}{Ordered integer vector of recommended item ids.}
\item{target}{Target item id.}
\item{kk}{Cutoff.}
}
\description{
1 if the target is in the top \code{k} ranked items, else 0.
}
\value{
0 or 1.
}
\keyword{internal}
\examples{
morie_gru4r_recall(ranked = c(1, 2, 3, 4, 5, 6, 7, 8), target = 5L)
}
