% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.fairness_backend}
\alias{.fairness_backend}
\title{Prefer native R torch; fall back to reticulate + JAX}
\usage{
.fairness_backend()
}
\description{
A step of the fairness_gan implementation. Called by \code{morie_fairness_ctgan_debiaser}, \code{morie_fairness_spatial_gan}. See the file header for the source the module follows.
}
\value{
A list with \code{kind}, \code{note}.
}
\keyword{internal}
\examples{
.fairness_backend()
}
