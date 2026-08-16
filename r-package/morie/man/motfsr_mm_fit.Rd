% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{motfsr_mm_fit}
\alias{motfsr_mm_fit}
\title{motfsr_mm_fit}
\usage{
motfsr_mm_fit(sequences, w, alphabet = NULL, theta0 = NULL, lambda0 = NULL, beta = 0.01, erasing = NULL, max_iter = 1000, tol = 1e-06, normalize_overlaps = TRUE, erase_by = "letter")
}
\arguments{
\item{sequences}{See Usage.}
\item{w}{See Usage.}
\item{alphabet}{Defaults to \code{NULL}.}
\item{theta0}{Defaults to \code{NULL}.}
\item{lambda0}{Defaults to \code{NULL}.}
\item{beta}{Defaults to \code{0.01}.}
\item{erasing}{Defaults to \code{NULL}.}
\item{max_iter}{Defaults to \code{1000}.}
\item{tol}{Defaults to \code{1e-06}.}
\item{normalize_overlaps}{Defaults to \code{TRUE}.}
\item{erase_by}{Defaults to \code{"letter"}.}
}
\description{
Part of the motfsr_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{theta}, \code{motif}, \code{background}, \code{lambda1}, \code{z}, \code{log_likelihood}, \code{log_likelihood_trace}, \code{n_iter}, \code{converged}, \code{alphabet}, \code{w}.
}
\references{
  Bailey & Elkan (1994) ISMB-94, 28-36.
  Bailey & Elkan (1995) ISMB-95, 21-29.
Base R only.
}
\keyword{internal}
