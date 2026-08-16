% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{sample_pdf}
\alias{sample_pdf}
\title{sample_pdf}
\usage{
sample_pdf(bins, weights, n_samples, rng, eps = 1e-05)
}
\arguments{
\item{bins}{Passed to \code{.as_numeric_vec}.}
\item{weights}{Passed to \code{.as_numeric_vec}.}
\item{n_samples}{Coerced to integer by the body, with \code{as.integer}.}
\item{rng}{Passed to \code{.ghc_unif}.}
\item{eps}{Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{1e-05}.}
}
\description{
Part of the nrfrad_native implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{sort}.
}
\keyword{internal}
