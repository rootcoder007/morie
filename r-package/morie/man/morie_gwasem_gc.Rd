% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gwasem_gc}
\alias{morie_gwasem_gc}
\title{Genomic control inflation factor}
\usage{
morie_gwasem_gc(stats, df = 1)
}
\arguments{
\item{stats}{Numeric vector of test statistics.}
\item{df}{Degrees of freedom (1 for chi-square, k for F).}
}
\description{
The ratio of the median test statistic to its null median.
}
\value{
Scalar lambda.
}
\keyword{internal}
