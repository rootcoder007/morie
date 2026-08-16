% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.mor_ks_2samp}
\alias{.mor_ks_2samp}
\title{Two-sample Kolmogorov-Smirnov statistic and p-value}
\usage{
.mor_ks_2samp(a, b)
}
\arguments{
\item{a}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{b}{Coerced to numeric by the body, with \code{as.numeric}.}
}
\description{
Part of the over_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{stat}, \code{p}, \code{n1}, \code{n2}, \code{d_plus}, \code{d_minus}, \code{n_ties}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.mor_ks_2samp(V, V)
}
