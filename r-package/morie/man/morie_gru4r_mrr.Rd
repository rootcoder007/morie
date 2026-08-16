% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gru4r_mrr}
\alias{morie_gru4r_mrr}
\title{Mean reciprocal rank at k}
\usage{
morie_gru4r_mrr(ranked, target, kk = 20)
}
\arguments{
\item{ranked}{Ordered integer vector of recommended item ids.}
\item{target}{Target item id.}
\item{kk}{Cutoff.}
}
\description{
1 / position of the target within the top k, or 0 if absent.
}
\value{
Scalar.
}
\keyword{internal}
\examples{
morie_gru4r_mrr(ranked = c(1, 2, 3, 4, 5, 6, 7, 8), target = 5L)
}
