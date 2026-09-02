% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_trend_vector}
\alias{morie_trend_vector}
\title{The trend vector T = (1/N) sum (a_i - A) S_i and its randomisation test}
\usage{
morie_trend_vector(torsion_sets, activities, permutations = 40, seed = 0) 
}
\arguments{
\item{torsion_sets}{List of torsion dictionaries.}
\item{activities}{Numeric activity vector.}
\item{permutations}{Number of permutations.}
\item{seed}{Seed for the shared generator.}
}
\description{
The trend vector T = (1/N) sum (a_i - A) S_i and its randomisation test
}
\value{
A list with the vector, descriptors, length, null mean and   sd, and the z-score.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
morie_trend_vector(V, V)
}
