% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.otis_predict_ps}
\alias{.otis_predict_ps}
\title{Predict propensity from fitted beta on a (possibly new) X}
\usage{
.otis_predict_ps(X, beta, eps = 0.02)
}
\arguments{
\item{X}{A matrix; passed to \code{\%*\%}.}
\item{beta}{A matrix; passed to \code{\%*\%}.}
\item{eps}{Passed to \code{.otis_clip_ps}. Defaults to \code{0.02}.}
}
\description{
Part of the otis_causal implementation; see the file header for the source it follows.
}
\value{
The value of \code{.otis_clip_ps}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.otis_predict_ps(V, V)
}
