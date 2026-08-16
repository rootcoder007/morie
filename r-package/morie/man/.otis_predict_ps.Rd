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
\item{X}{See Usage.}
\item{beta}{See Usage.}
\item{eps}{Defaults to \code{0.02}.}
}
\description{
Part of the otis_causal implementation; see the file header for the source it follows.
}
\value{
The value of \code{.otis_clip_ps}.
}
\keyword{internal}
