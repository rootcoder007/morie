% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{lemma4_bound}
\alias{lemma4_bound}
\title{Lemma 4: sum_{j != j*} (2 + gamma (n_j* - n_j)) /}
\usage{
lemma4_bound(counts, gamma)
}
\arguments{
\item{counts}{See Usage.}
\item{gamma}{See Usage.}
}
\description{
(4 exp(gamma (n_j* - n_j))), j* the plurality winner. The bound is a bound, so the raw value is also returned.
}
\value{
A list with \code{bound}, \code{raw}.
}
\keyword{internal}
\examples{
lemma4_bound(counts = c(1, 2, 3, 4, 5, 6, 7, 8), gamma = 0.5)
}
