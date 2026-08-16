% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.ghebp_negll}
\alias{.ghebp_negll}
\title{Internal: Antoniak DP empirical-Bayes negative log-likelihood in the}
\usage{
.ghebp_negll(a, K_n, n)
}
\arguments{
\item{a}{Numeric; passed to \code{log}.}
\item{K_n}{Numeric; combined arithmetically in the body.}
\item{n}{Numeric; combined arithmetically in the body.}
}
\description{
concentration parameter alpha. Extracted from the morie_ghosal_empirical_bayes() optimiser closure for direct unit-testing.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
.ghebp_negll(a = c(1, 2, 3, 4, 5, 6, 7, 8), K_n = c(1, 2, 3, 4, 5, 6, 7, 8), n = 5L)
}
